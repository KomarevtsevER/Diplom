import cv2
import time

# Указываем путь к изображению
image_path = '/Users/komarevtsev/Documents/Diplom/MyDataset/Data/imag/146.jpg'

# Загружаем веса и модели нейросетей
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"


# Функция определения лиц
def highlightFace(net, frame, conf_threshold=0.7):
    # Делаем копию текущего кадра
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]

    # Преобразуем картинку в двоичный пиксельный объект
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False  , False)

    # Устанавливаем этот объект как входной параметр для нейросети
    net.setInput(blob)

    # Выполняем прямой проход для распознавания лиц
    detections = net.forward()

    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            # Рисуем рамку
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)

            # Выводим координаты в консоль
            print(f"Левая верхняя точка: ({x1}, {y1}), Правая нижняя точка: ({x2}, {y2})")

            # Добавляем текст с координатами на изображение
            cv2.putText(frameOpencvDnn, f'({x1},{y1}) - ({x2},{y2})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frameOpencvDnn, faceBoxes


# Настроим значения
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Загружаем нейросети
faceNet = cv2.dnn.readNet(faceModel, faceProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)

# Загружаем изображение с помощью OpenCV
frame = cv2.imread(image_path)

# Проверяем, загрузилось ли изображение
if frame is None:
    print("Ошибка: Не удалось загрузить изображение.")
    exit()

# Распознаем лица в изображении и измеряем время
start_time = time.time()
resultImg, faceBoxes = highlightFace(faceNet, frame)
face_detection_time = time.time() - start_time

if not faceBoxes:
    print("Лица не распознаны")
else:
    print(f"Найдено лиц: {len(faceBoxes)}")
    print(f"Время распознавания лиц: {face_detection_time:.4f} секунд.")

# Обрабатываем каждое лицо, предсказываем пол и возраст
for faceBox in faceBoxes:
    # Получаем изображение лица на основе рамки
    face = frame[max(0, faceBox[1]):min(faceBox[3], frame.shape[0] - 1), max(0, faceBox[0]):min(faceBox[2], frame.shape[1] - 1)]
    # Преобразуем изображение в бинарный формат для нейросети
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB = False)

    # Определяем пол и измеряем время
    start_time = time.time()
    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender_time = time.time() - start_time

    gender = genderList[genderPreds[0].argmax()]
    print(f'Gender: {gender} (время определения: {gender_time:.4f} секунд)')

    # Определяем возраст и измеряем время
    start_time = time.time()
    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age_time = time.time() - start_time

    age = ageList[agePreds[0].argmax()]
    print(f'Age: {age[1:-1]} years (время определения: {age_time:.4f} секунд)')

    # Добавляем текст возле каждой рамки в кадре
    cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 255), 2, cv2.LINE_AA)

# Отображаем окно с изображением
cv2.imshow("Detecting Age and Gender", resultImg)

# Ожидаем закрытия окна
cv2.waitKey(0)
cv2.destroyAllWindows()

