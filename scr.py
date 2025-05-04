import os
import re
import numpy as np
import cv2
import pytesseract
import pandas as pd
import json
from natasha import Segmenter, NewsEmbedding, NewsNERTagger, Doc
from transliterate import translit
from deskew import determine_skew

# Функции предобработки изображения
def load_image(input_filename):
    """Загрузка изображения"""
    image = cv2.imread(input_filename)
    if image is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение по пути: {input_filename}")
    #Изменение размеров изображения
    image=cv2.resize(image,(640,640))
    print(f"Изображение успешно загружено: {input_filename}")
    return image

def processing(image):
    """Предобработка изображения для лучшей работы pytesseract"""
    # Удаление шума (Denoising)

    denoised = cv2.GaussianBlur(image, (3, 3), 0)  # ядро 3x3
    # 4.2. Усиливаем резкость (опционально)
    kernel = np.array([[0, -1, 0],
                        [-1, 5,-1],
                        [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    # Пороговая обработка (Thresholding)
    _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_OTSU) # Метод Оцу для автоматического определения порога

    # Выпрямление документа по главной оси (Deskew)
    angle = determine_skew(thresh)
    (h, w) = thresh.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)  # отрицательный угол для корректного поворота
    deskewed = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return deskewed

def rotate(image,angle):
    """Поворот изображения на 90 градусов по часовой и против часовой стрелки."""
    if angle == 90:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == -90:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

#Функции распознавания текста и фильтрации текста
def recognition(image):
    """Распознавание надписей на кириллице и латинице на изображении"""
    data = pytesseract.image_to_data(
        image,
        lang='rus',    # или 'eng' если английский, или 'rus+eng' для обоих
        output_type=pytesseract.Output.DATAFRAME
    )

    # Распознавание с возвратом данных о расположении
    data_eng = pytesseract.image_to_data(
        image,
        lang='rus+eng',    # или 'eng' если английский, или 'rus+eng' для обоих
        output_type=pytesseract.Output.DATAFRAME
    )

    # Заменяем текст в колонке на транслитерированный
    data_eng['text'] = data_eng['text'].apply(lambda x: translit(x, 'ru') if type(x) is str else x)
    data_sum=pd.concat([data,data_eng],ignore_index=True)
    # Фильтрация пустых строк
    data_sum = data_sum.dropna(subset=['text'])
    data_sum = data_sum[data_sum['text'].apply(lambda x: isinstance(x, str) and x.strip() != '')]

    # убираем строки с одинаковым текстом и координатами
    data_sum = filter_dataframe(data_sum)
    data_sum['text'] = data_sum['text'].astype(str)


    return data_sum


def filter_dataframe(data):
       """Фильтрует DataFrame по заданным условиям."""
       data = data[~data.duplicated(subset=['text'], keep='first')]
       data = data[~data.duplicated(subset=['left', 'top'], keep='first')]
       return data.reset_index(drop=True)


def filter_names_gender_place(data):
    """Фильтрация датафрейма для последующего поиска пола, ФИО и места рождения"""
    #Фильтруем те строки, расстояние между которыми по вертикале меньше 10 пикселей
    # Сортируем по top
    data = data.sort_values('top').reset_index(drop=True)

    # Фильтруем по координатам
    data = data[(data['left'] > 100) & (data['left'] < 540)]
    data = data[(data['top'] > 30) & (data['top'] < 500)]
    data=data.reset_index(drop=True)

    # Оставляем только русские буквы (удаляем всё, что не кириллица)
    data['text'] = data['text'].apply(lambda x: re.sub(r'[^а-яА-Я]', '', str(x).capitalize()))

    # Список для индексов, которые надо оставить
    filtered_indices = []

    # Начинаем с первой строки
    last_accepted = None

    #Фильтруем строки, разница по у между которыми меньше 10 пикселей, чтобы убрать одинаковые сущности
    for idx, row in data.iterrows():
        current_top = row['top']
        if last_accepted is None or abs(current_top - last_accepted) > 10:
            filtered_indices.append(idx)
            last_accepted = current_top


    data=data.reset_index(drop=True)
    return data



#Функции распознавания сущностей
def extract_ner(text):
    """Функция для выделения сущностей"""
    doc = Doc(text.capitalize())
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    return [(span.text, span.type) for span in doc.spans]

def dates_recognition(data):
    """Обнаружение даты рождения на изображении"""
    # Регулярное выражение для поиска даты
    pattern = r'\d{2}\.\d{2}\.\d{4}'

    dates = data[data['text'].str.match(pattern)]
    
    # убираем строки с одинаковым текстом и координатами
    dates = filter_dataframe(dates)

    dates_coord = (dates['left'][0], dates['top'][0], dates['width'][0], dates['height'][0])

    return dates_coord

def series_number_recognition(data):
    """Обнаружение серии и номера паспорта"""

    series_number = data[data['text'].apply(lambda x: x.isdigit())] #оставляем только строки с цифрами
    series= series_number[series_number['text'].apply(lambda x: len(x)==2)] #оставляем строки только с 2 цифрами (серия)

    number=series_number[series_number['text'].apply(lambda x: len(x)>=4)] ##оставляем строки только с количеством цифр больше 4 (номер)

    number = number.reset_index(drop=True)

    series = series.reset_index(drop=True)

    series_coords=[]
    # Проходим по всем строкам с серии
    for _, row in series.iterrows():
        tech= (int(row['left']), int(row['top']), int(row['width']), int(row['height']))
        series_coords.append(tech)

    if not number.empty:
        numbers = (number['left'][0], number['top'][0], number['width'][0], number['height'][0])
    else:
        numbers=[]

    return series_coords,numbers


def gender_recognition(data):
    """Обнаружения поля пола на изображении"""
    # Превращаем все буквы в маленькие
    data['text'] = data['text'].apply(lambda x: re.sub(r'[^а-яА-Я]', '', str(x).lower()))
    #паттерн поиска
    pattern = r"^(м|уж|ж|ен|муж|жен)(у|е|уж|ен)?$" 
    data = data[data['text'].str.match(pattern)]

    # убираем строки с одинаковым текстом и координатами
    data = filter_dataframe(data)

    # Создаем новую колонку, где подсчитываем количество букв в каждой строке
    data['letter_count'] = data['text'].apply(lambda x: len(x)<=3 and len(re.findall(r'[a-zA-Zа-яА-Я]', str(x))))
    data = data.reset_index(drop=True)
    # Находим строку с максимальным количеством букв
    if not data.empty:
        data = data.loc[[data['letter_count'].idxmax()]]

        data=data.reset_index(drop=True)

        gender_coord = (data['left'][0], data['top'][0], data['width'][0], data['height'][0])

        return gender_coord
    else:
        return []

def photo_recognition(image):
    """Обнаружение фотографии на изображении"""
    # Обнаруживаем лица на изображении
    faces = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    
    if faces.any():
        # Найдём лицо с минимальной координатой x (самое левое)
        leftmost_face = min(faces, key=lambda x: x[0])

        photo_coord = (leftmost_face[0]-10,leftmost_face[1]-50,leftmost_face[2]+10,leftmost_face[3]+100)

    return photo_coord

def name_place_recognition(nameplace):
    """Фильтрация ФИО и места рождения из датафрейма"""
    # Применение функции extract_ner к столбцу 'text'
    nameplace['entities'] = nameplace['text'].apply(extract_ner)
    # Фильтруем строки, где были найдены сущности
    nameplace = nameplace.dropna(subset=['entities'])  # Убираем строки с None в сущностях
    nameplace = nameplace[nameplace['entities'].apply(lambda x: len(x) > 0)]  # Оставляем строки, где есть сущности
    #Только LOC
    place = nameplace[nameplace['entities'].apply(lambda ents: any(label == 'LOC' for _, label in ents))]
    place=place.reset_index(drop=True)
    place_coords=[]
    # Проходим по всем строкам с LOC
    for _, row in place.iterrows():
        tech= (int(row['left']), int(row['top']), int(row['width']), int(row['height']))
        place_coords.append(tech)
        
    # Копия только строк с PER
    names = nameplace[nameplace['entities'].apply(lambda ents: any(label == 'PER' for _, label in ents))]
    names=names.reset_index(drop=True)

    names_coords=[]
    # Проходим по всем строкам с PER
    for _, row in names.iterrows():
        tech= (int(row['left']), int(row['top']), int(row['width']), int(row['height']))
        names_coords.append(tech)

    return place_coords, names_coords

def surname_name_otchestvo(names_coords,dates_coord,gender_coord,coord1,coord2):
    """Ищет имя, фамилию и отчество на изображении. dates_coord,gender_coord нужны для определения, где имя, где фамилия, 
    а где отчество относительно поля даты рождения или пола через сравнения разности координат по y через вычисления разности 
    и сравнения с интервалом [coord1,coord2]"""
    check=[]
    if dates_coord[1]:
        check=[dates_coord[1]] #координаты y полей даты рождения и пола для последующей проверки полей ФИО
    elif gender_coord[1]:
        check=[gender_coord[1]] #координаты y полей даты рождения и пола для последующей проверки полей ФИО
    name_coord=[]
    # Проходим по всем строкам с ФИО
    for x,y,w,h in names_coords:
        if check:
            for coord in check: #проверяем поля ФИО относительно полей даты рождения и пола и интервала разницы [coord1,coord2] для определения имени, фамилии и отчества
                if coord:
                    if coord1<=coord-y<=coord2:
                        tech=[x, y,w,h]
                        name_coord.append(tech)
    
    if name_coord:
        return name_coord[0]
    else:
        return name_coord



#Разметка изображения
def sign(image,coord,text):
    """Помечает bounding boxes на изображении по координатам"""
    if coord:
        x,y,w,h=coord
        # Рисуем прямоугольник
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Добавляем подпись
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3, cv2.LINE_AA)  # красный текст
    return image



#Функции создания JSON COCO
def convert_numpy_types(obj):
    """Рекурсивно приводит все numpy-типы к стандартным Python-типа для сериализации."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif hasattr(obj, 'item'): 
        return obj.item()
    else:
        return obj

def save_image_and_json(image, coco_output, image_path):
    """
    Сохраняет изображение и JSON-файл в папку 'output', находящуюся в той же папке, что и исходное изображение.
    """
    # Получаем директорию исходного изображения
    base_dir = os.path.dirname(image_path)
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Получаем имя файла без расширения
    filename = os.path.splitext(os.path.basename(image_path))[0]

    # Пути для сохранения
    image_save_path = os.path.join(output_dir, f"{filename}.jpg")
    json_save_path = os.path.join(output_dir, f"{filename}.json")

    # Сохраняем изображение
    cv2.imwrite(image_save_path, image)

    # Приводим numpy-типы к обычным
    cleaned_output = convert_numpy_types(coco_output)

    # Сохраняем JSON
    with open(json_save_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_output, f, ensure_ascii=False, indent=4)






# Настройка Natasha
emb = NewsEmbedding()
segmenter = Segmenter()
ner_tagger = NewsNERTagger(emb)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Загружаем предобученный классификатор лиц
cascade_path = os.path.join(BASE_DIR, 'haarcascades', 'haarcascade_frontalface_default.xml')
detector = cv2.CascadeClassifier(cascade_path)
#путь до исполняемого файла Tesseract OCR
tesseract_path = os.path.join(BASE_DIR, 'tesseract', 'tesseract.exe')
pytesseract.pytesseract.tesseract_cmd = tesseract_path


#Загрузка изобаржения
input_filename = input("Введите путь к изображению: ")
image=load_image(input_filename)

# Примерные данные
file_name = input_filename.split('/')[-1]
image_width = image_height = 640
image_id = 1

# Перевод в градации серого (Grayscale)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

processed=processing(gray) #предобработка

crop=rotate(processed,-90)[0:100] #предобработка изображения серии и номера


data_rotated=recognition(crop) #распознавание текста серии и номера


data=recognition(processed) #распознавание текста на изображении

data_names_gender_place=filter_names_gender_place(data) #фильтрация из распознанного текста имени, пола и места рождения

series_coords, number_coord=[],[]

if not data_rotated.empty:

    series_coords, number_coord= series_number_recognition(data_rotated) #поиск координатов серии и номера паспорта

dates_coord=dates_recognition(data) #поиск координатов даты рождения

photo_coord=photo_recognition(gray) #поиск координатов фото

gender_coord=gender_recognition(data) #поиск координотов пола

place_coords, names_coords=name_place_recognition(data_names_gender_place) #поиск координатов места рождения и ФИО

surname_coord=surname_name_otchestvo(names_coords,dates_coord,gender_coord,130,220) #поиск фамилии
name_coord=surname_name_otchestvo(names_coords,dates_coord,gender_coord,80,120) #поиск имени
otchestvo_coord=surname_name_otchestvo(names_coords,dates_coord,gender_coord,30,70) #поиск отчества

image=rotate(image,-90) #поворот исходного изображения на 90 градусов против часовой

image=sign(image,number_coord,'number') #разметка номера

for coord in series_coords: #разметка серии
    image=sign(image,coord,'ser')

image=rotate(image,90) #возврат изображения в исходную позицию

for coord in place_coords: #разметка места рождения
    image=sign(image,coord,'place')

cycle_annot = [
    (dates_coord, 'dates'),
    (photo_coord, 'photo'),
    (gender_coord, 'gender'),
    (surname_coord, 'surname'),
    (name_coord, 'name'),
    (otchestvo_coord, 'otchestvo')
]

if not (surname_coord or name_coord or otchestvo_coord):
    cycle_annot = [
    (dates_coord, 'dates'),
    (photo_coord, 'photo'),
    (gender_coord, 'gender'),
    ]
    
    for coord in names_coords: #разметка места рождения
        image=sign(image,coord,'FIO')

for coord, text in cycle_annot: #разметка остальнвх полей с помощью словаря и цикла
    image = sign(image, coord, text)




#Перевод аннотаций в JSON COCO
# Категории
categories = [
    "photo", "surname", "name", "otchestvo", "gender", "dates",
    "place", "ser", "number"
]

# Словарь категорий для ID
category_map = {name: idx + 1 for idx, name in enumerate(categories)}

# Собираем все аннотации
annotation_id = 1
annotations = []

def add_annotation(bbox, category_name):
    global annotation_id
    annotations.append({
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_map[category_name],
        "bbox": list(bbox),
        "iscrowd": 0
    })
    annotation_id += 1


for coord, text in cycle_annot: #запись в json разметки полей с помощью словаря и цикла
    add_annotation(coord, text)

for coord in place_coords:
    add_annotation(coord, "place")
for coord in series_coords:
    add_annotation(coord, "ser")

# Итоговый COCO JSON
coco_output = {
    "images": [{
        "id": image_id,
        "file_name": file_name,
        "width": image_width,
        "height": image_height
    }],
    "annotations": annotations,
    "categories": [
        {"id": cid, "name": name} for name, cid in category_map.items()
    ]
}


save_image_and_json(image, coco_output, input_filename) #сохранение json файла





# Показываем изображение
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
