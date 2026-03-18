import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import random
import re
import os
import time

# Пути к моделям
faceProto   = "opencv_face_detector.pbtxt"
faceModel   = "opencv_face_detector_uint8.pb"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
ageProto    = "age_deploy.prototxt"
ageModel    = "age_net.caffemodel"

# Загрузка моделей
faceNet   = cv2.dnn.readNet(faceModel, faceProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
ageNet    = cv2.dnn.readNet(ageModel, ageProto)

# Настройки
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']
ageList    = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Счётчики времени
fd_time, fd_count = 0.0, 0
gen_time, gen_count = 0.0, 0
age_time, age_count = 0.0, 0

def load_custom_annotations(label_file, images_dir):
    annotations = {}
    with open(label_file, "r") as file:
        lines = file.readlines()

    image_path = None
    faces = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if image_path:
                annotations[image_path] = faces
            image_path = f"{images_dir}/{line[2:]}"
            faces = []
        else:
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                x1, y1, x2, y2 = map(int, parts[:4])
            except ValueError:
                continue
            gender = parts[4]
            age    = parts[5]
            faces.append((x1, y1, x2, y2, gender, age))
    if image_path:
        annotations[image_path] = faces
    return annotations

def detect_faces(frame):
    global fd_time, fd_count
    start = time.perf_counter()

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 [104, 117, 123], swapRB=True)
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > 0.7:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            faces.append((x1, y1, x2, y2))

    fd_time += time.perf_counter() - start
    fd_count += 1
    return faces

def predict_gender_age(face):
    global gen_time, gen_count, age_time, age_count

    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                 MODEL_MEAN_VALUES, swapRB=False)

    # Gender
    start_g = time.perf_counter()
    genderNet.setInput(blob)
    genderPred = genderNet.forward()[0]
    gen_time += time.perf_counter() - start_g
    gen_count += 1
    gender = genderList[np.argmax(genderPred)]

    # Age
    start_a = time.perf_counter()
    ageNet.setInput(blob)
    agePred = ageNet.forward()[0]
    age_time += time.perf_counter() - start_a
    age_count += 1
    age = ageList[np.argmax(agePred)]

    return gender, age

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if boxAArea + boxBArea - interArea == 0:
        return 0.0
    return interArea / float(boxAArea + boxBArea - interArea)

def evaluate_face_detection(pred_faces, true_faces, iou_threshold=0.5):
    matches = sum(
        1 for pred in pred_faces
        if any(iou(pred, tf[:4]) > iou_threshold for tf in true_faces)
    )
    return matches / max(len(true_faces), 1)

def evaluate_custom_model(images_dir, label_file):
    annotations = load_custom_annotations(label_file, images_dir)
    image_paths = list(annotations.keys())

    y_true_gender, y_pred_gender = [], []
    y_true_age, y_pred_age       = [], []
    iou_scores = []

    for image_path in image_paths:
        true_faces = annotations[image_path]
        frame = cv2.imread(image_path)
        if frame is None:
            continue

        pred_faces = detect_faces(frame)
        iou_scores.append(evaluate_face_detection(pred_faces, true_faces))

        for x1,y1,x2,y2,true_gender,true_age in true_faces:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            pred_gender, pred_age = predict_gender_age(crop)
            y_true_gender.append(true_gender)
            y_pred_gender.append(pred_gender)
            y_true_age.append(true_age)
            y_pred_age.append(pred_age)

    face_detection_rate = sum(iou_scores) / len(iou_scores)
    print(f"\nДоля корректно детектированных лиц (IoU > 0.5): {face_detection_rate:.2%}\n")

    print("Отчёт по классификации пола:")
    print(classification_report(y_true_gender, y_pred_gender, zero_division=0))

    print("Отчёт по классификации возраста:")
    print(classification_report(y_true_age, y_pred_age, zero_division=0))

    print("\n=== Сводка по времени работы ===")
    if fd_count:
        print(f"Обнаружение лиц: {fd_time:.3f} сек за {fd_count} вызов(ов), среднее: {fd_time/fd_count:.3f} сек")
    if gen_count:
        print(f"Модель определения пола: {gen_time:.3f} сек за {gen_count} вызов(ов), среднее: {gen_time/gen_count:.3f} сек")
    if age_count:
        print(f"Модель определения возраста: {age_time:.3f} сек за {age_count} вызов(ов), среднее: {age_time/age_count:.3f} сек")

if __name__ == "__main__":
    random.seed(42)
    images_dir = "Data/imag"
    label_file = "Data/annot.txt"
    evaluate_custom_model(images_dir, label_file)
