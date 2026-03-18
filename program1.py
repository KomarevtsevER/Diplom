import cv2
import os
import time

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
haarCascade = "haarcascade_frontalface_default.xml"

IOU_THRESHOLD = 0.5

# Временные метки
haar_time = 0.0
haar_calls = 0
dnn_time = 0.0
dnn_calls = 0

haar_custom_time = 0.0
haar_custom_calls = 0
dnn_custom_time = 0.0
dnn_custom_calls = 0

haar_clf = cv2.CascadeClassifier(haarCascade)
dnn_net = cv2.dnn.readNetFromTensorflow(faceModel, faceProto)


def detect_faces_haar(img, is_custom=False):
    global haar_time, haar_calls, haar_custom_time, haar_custom_calls
    start = time.perf_counter()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = haar_clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    elapsed = time.perf_counter() - start
    if is_custom:
        haar_custom_time += elapsed
        haar_custom_calls += 1
    else:
        haar_time += elapsed
        haar_calls += 1

    return [(x, y, x + w, y + h) for (x, y, w, h) in rects]


def detect_faces_dnn(img, conf_thresh=0.7, is_custom=False):
    global dnn_time, dnn_calls, dnn_custom_time, dnn_custom_calls
    start = time.perf_counter()

    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], swapRB=True, crop=False)
    dnn_net.setInput(blob)
    detections = dnn_net.forward()

    elapsed = time.perf_counter() - start
    if is_custom:
        dnn_custom_time += elapsed
        dnn_custom_calls += 1
    else:
        dnn_time += elapsed
        dnn_calls += 1

    boxes = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf > conf_thresh:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            boxes.append((x1, y1, x2, y2))
    return boxes


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - interArea
    return interArea / union if union > 0 else 0


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


def load_custom_annotations(annot_txt, img_root):
    gt = {}
    current = None
    with open(annot_txt, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                fname = line[1:].strip()
                current = os.path.join(img_root, fname)
                gt[current] = []
            else:
                parts = line.split()
                if len(parts) >= 4 and parts[0].isdigit():
                    x1, y1, x2, y2 = map(int, parts[:4])
                    gt[current].append((x1, y1, x2, y2))
    return gt


def evaluate_detection(detect_fn, gt_dict, is_custom=False):
    scores = []
    for img_path, true_boxes in gt_dict.items():
        img = cv2.imread(img_path)
        if img is None or not true_boxes:
            continue
        pred_boxes = detect_fn(img, is_custom=is_custom)
        matched = 0
        for tb in true_boxes:
            if any(iou(tb, pb) >= IOU_THRESHOLD for pb in pred_boxes):
                matched += 1
        scores.append(matched / len(true_boxes))
    return sum(scores) / len(scores) if scores else 0


if __name__ == "__main__":
    fddb_txt = "Dataset_FDDB/label.txt"
    fddb_img_root = "Dataset_FDDB/images"
    custom_txt = "Data/annot.txt"
    custom_img_root = "Data/imag"

    fddb = load_fddb_annotations(fddb_txt, fddb_img_root)
    custom = load_custom_annotations(custom_txt, custom_img_root)

    print("=== Точность на датасете FDDB ===")
    print(f"Метод каскадов Хаара: {evaluate_detection(detect_faces_haar, fddb):.2%}")
    print(f"Глубокие нейронные сети:  {evaluate_detection(detect_faces_dnn, fddb):.2%}")

    print("\n=== Точность на персональном датасете ===")
    print(f"Метод каскадов Хаара: {evaluate_detection(detect_faces_haar, custom, is_custom=True):.2%}")
    print(f"Глубокие нейронные сети:  {evaluate_detection(detect_faces_dnn, custom, is_custom=True):.2%}")

    print("\n=== Сводка по времени работы ===")
    if haar_calls:
        print(f"Метод каскадов Хаара на FDDB: {haar_time:.3f} сек за {haar_calls} вызовов, среднее: {haar_time / haar_calls:.3f} сек")
    if dnn_calls:
        print(f"Метод глубоких нейронных сетей на FDDB: {dnn_time:.3f} сек за {dnn_calls} вызовов, среднее: {dnn_time / dnn_calls:.3f} сек")
    if haar_custom_calls:
        print(f"Метод каскадов Хаара на персональном датасете: {haar_custom_time:.3f} сек за {haar_custom_calls} вызовов, среднее: {haar_custom_time / haar_custom_calls:.3f} сек")
    if dnn_custom_calls:
        print(f"Метод глубоких нейронных сетей на персональном датасете: {dnn_custom_time:.3f} сек за {dnn_custom_calls} вызовов, среднее: {dnn_custom_time / dnn_custom_calls:.3f} сек")
