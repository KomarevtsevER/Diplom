import cv2
import numpy as np
import os
import random
import time
from sklearn.metrics import accuracy_score, classification_report

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

# Константы
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']
ageList    = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
              '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Счётчики времени
fd_time = 0.0
fd_count = 0
gen_time = 0.0
gen_count = 0
age_time = 0.0
age_count = 0

def normalize_age_label(age_str):
    try:
        low, high = map(int, age_str.strip('()').split(','))
        for label in ageList:
            a, b = map(int, label.strip('()').split('-'))
            if low >= a and high <= b:
                return label
    except:
        pass
    return None

def load_adience_annotations(label_file):
    ann = {}
    if not os.path.exists(label_file):
        print(f"⚠ Файл {label_file} не найден!")
        return ann
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = [p for p in line.strip().split('\t') if p]
            if len(parts) < 5:
                continue
            folder, img_name, num = parts[0], parts[1], parts[2]
            age_raw, gender_raw = parts[3], parts[4].lower()
            if gender_raw not in ('m','f'):
                continue
            age_norm = normalize_age_label(age_raw)
            if age_norm is None:
                continue
            path = os.path.join('faces', folder,
                                f'coarse_tilt_aligned_face.{num}.{img_name}')
            ann[path] = {
                'gender': 'Male' if gender_raw=='m' else 'Female',
                'age':    age_norm
            }
    return ann

def load_fddb_annotations(label_file, images_dir):
    ann = {}
    with open(label_file, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    img_path = None
    faces = []
    for line in lines:
        if line.startswith('#'):
            if img_path:
                ann[img_path] = faces
            img_path = os.path.join(images_dir, line[2:])
            faces = []
        else:
            coords = list(map(int, line.split()))
            faces.append(tuple(coords))
    if img_path:
        ann[img_path] = faces
    return ann

def detect_faces(frame, conf_threshold=0.8, nms_threshold=0.4, min_size=50):
    global fd_time, fd_count
    start = time.perf_counter()
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300),
                                 [104,117,123], swapRB=False)
    faceNet.setInput(blob)
    dets = faceNet.forward()

    boxes, confs = [], []
    for i in range(dets.shape[2]):
        conf = float(dets[0,0,i,2])
        if conf < conf_threshold:
            continue
        x1 = int(dets[0,0,i,3]*w)
        y1 = int(dets[0,0,i,4]*h)
        x2 = int(dets[0,0,i,5]*w)
        y2 = int(dets[0,0,i,6]*h)
        if (x2-x1) < min_size or (y2-y1) < min_size:
            continue
        boxes.append([x1, y1, x2-x1, y2-y1])
        confs.append(conf)

    idxs = cv2.dnn.NMSBoxes(boxes, confs, conf_threshold, nms_threshold)
    faces = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, bw, bh = boxes[i]
            faces.append((x, y, x+bw, y+bh))

    fd_time += time.perf_counter() - start
    fd_count += 1
    return faces

def predict_gender_age(face):
    global gen_time, gen_count, age_time, age_count
    blob = cv2.dnn.blobFromImage(face, 1.0, (227,227),
                                 MODEL_MEAN_VALUES, swapRB=False)
    start_g = time.perf_counter()
    genderNet.setInput(blob)
    gpred = genderNet.forward()[0]
    gen_time += time.perf_counter() - start_g
    gen_count += 1

    start_a = time.perf_counter()
    ageNet.setInput(blob)
    apred = ageNet.forward()[0]
    age_time += time.perf_counter() - start_a
    age_count += 1

    gender = genderList[np.argmax(gpred)]
    age = ageList[np.argmax(apred)]
    return gender, age

def evaluate_fddb_model(images_dir, label_file, iou_thr=0.5):
    ann = load_fddb_annotations(label_file, images_dir)
    rates = []
    for path, true_faces in ann.items():
        img = cv2.imread(path)
        if img is None or not true_faces:
            continue
        pred = detect_faces(img)
        matches = sum(1 for pf in pred
                      if any(iou(pf, tf) > iou_thr for tf in true_faces))
        rates.append(matches / len(true_faces))
    if rates:
        print(f"Точность обнаружения на FDDB при IoU ≥ {iou_thr}: {np.mean(rates):.2%}")
    else:
        print("Нет данных для оценки на FDDB.")

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB-xA)*max(0, yB-yA)
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    denom = areaA + areaB - inter
    return inter/denom if denom > 0 else 0

def evaluate_adience_model(label_file):
    ann = load_adience_annotations(label_file)
    y_true_g, y_pred_g = [], []
    y_true_a, y_pred_a = [], []

    for path, true in ann.items():
        img = cv2.imread(path)
        if img is None:
            continue
        faces = detect_faces(img)
        if not faces:
            continue

        x1,y1,x2,y2 = faces[0]
        h, w = img.shape[:2]
        x1, y1 = max(0,x1), max(0,y1)
        x2, y2 = min(w,x2), min(h,y2)
        if x2<=x1 or y2<=y1:
            continue

        face = img[y1:y2, x1:x2]
        if face.size == 0:
            continue

        gender, age = predict_gender_age(face)
        y_true_g.append(true['gender'])
        y_pred_g.append(gender)
        y_true_a.append(true['age'])
        y_pred_a.append(age)

    print(f"\nТочность определения пола (Adience): {accuracy_score(y_true_g, y_pred_g):.2%}")
    print("Отчёт по классификации пола:")
    print(classification_report(
        y_true_g, y_pred_g,
        target_names=genderList,
        zero_division=0
    ))

    idx = {lbl:i for i,lbl in enumerate(ageList)}
    one_off_count = sum(
        abs(idx[t] - idx[p]) <= 1
        for t, p in zip(y_true_a, y_pred_a)
    )
    one_off_acc = one_off_count / len(y_true_a)
    print(f"\nТочность определения возраста: {one_off_acc:.2%}")

    y_pred_relaxed = [
        t if abs(idx[t] - idx[p]) <= 1 else p
        for t, p in zip(y_true_a, y_pred_a)
    ]
    print("\nОтчёт по классификации возраста:")
    print(classification_report(
        y_true_a,
        y_pred_relaxed,
        labels=ageList,
        target_names=ageList,
        zero_division=0
    ))

if __name__ == "__main__":
    random.seed(42)

    print("=== Оценка точности на датасете FDDB ===")
    evaluate_fddb_model(
        images_dir="Dataset_FDDB/images",
        label_file="Dataset_FDDB/label.txt",
        iou_thr=0.5
    )

    print("\n=== Оценка точности на датасете Adience ===")
    evaluate_adience_model(
        label_file="fold_frontal_0_data.txt"
    )

    print("\n=== Сводка производительности ===")
    if fd_count:
        print(f"Обнаружение лиц: {fd_time:.3f} сек за {fd_count} вызов(ов), среднее: {fd_time/fd_count:.3f} сек")
    if gen_count:
        print(f"Модель определения пола: {gen_time:.3f} сек за {gen_count} вызов(ов), среднее: {gen_time/gen_count:.3f} сек")
    if age_count:
        print(f"Модель определения возраста: {age_time:.3f} сек за {age_count} вызов(ов), среднее: {age_time/age_count:.3f} сек")
