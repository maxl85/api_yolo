# Описане детектора
metadata = {
    "type": "yolo8s",
    "model": "/models/yolo_v8_model.pt",
    "dataset": "/datasets/yolo_v8_model/",
}

def predict(image):
    # Пример для YOLO v8
    model = YOLO("yolov8n.pt")
    
    # Распознаем
    results = model.predict(image)
    
    # Вытаскиваем рузельтаты распознавания
    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            detections.append({
                "class": model.names[class_id],
            })
    
    return detections
    
    
def train(detector_name, dataset_path): # del dataset_path
    # Тут должен быть код на котором тренировали модель
    
    result = {"detector_name": detector_name, "dataset_path": dataset_path}
    return result


def get_metadata(detector_name):
    return {'name': detector_name, **metadata}
