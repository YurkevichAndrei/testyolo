import cv2
import numpy as np
from ultralytics import YOLO


class ModelHandler:
    def __init__(self, labels):
        self.device = 'cpu'
        self.labels = labels
        self.load_network(model="yolo_rgb_weights.pt")

    def load_network(self, model):
        try:
            # Загрузка модели с помощью ultralytics.YOLO
            self.model = YOLO(model)
            self.model.to(self.device)
            self.is_inititated = True
        except Exception as e:
            raise Exception(f"Cannot load model {model}: {e}")

    def _infer(self, inputs: np.ndarray):
        try:
            # Inference с помощью YOLO
            results = self.model(inputs, augment=True)
            
            # Обработка результатов
            if len(results) == 0:
                return None, None, None

            print(results[0].to_json())
            
            result = results[0]  # Берем первый результат (для одного изображения)
            
            # Получаем bounding boxes
            if result.boxes is None:
                return None, None, None
                
            boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            scores = result.boxes.conf.cpu().numpy()  # confidence scores
            class_ids = result.boxes.cls.cpu().numpy().astype(int)  # class IDs

            print(boxes, scores, class_ids)

            return boxes, class_ids, scores

        except Exception as e:
            print(f"Inference error: {e}")
            return None, None, None

    def infer(self, image, threshold):
        image = np.array(image)
        # Конвертируем RGB в BGR для OpenCV (YOLO работает с RGB)
        image = image[:, :, ::-1].copy()  # RGB to BGR
        h, w, _ = image.shape
        
        boxes, class_ids, scores = self._infer(image)

        results = []
        if boxes is not None and len(boxes) > 0:
            for i, (box, class_id, score) in enumerate(zip(boxes, class_ids, scores)):
                if score >= threshold:
                    x1, y1, x2, y2 = box
                    
                    xtl = max(int(x1), 0)
                    ytl = max(int(y1), 0)
                    xbr = min(int(x2), w)
                    ybr = min(int(y2), h)

                    # Проверка валидности bounding box
                    if xbr > xtl and ybr > ytl:
                        results.append({
                            "confidence": str(score),
                            "label": self.labels.get(int(class_id), "unknown"),
                            "points": [xtl, ytl, xbr, ybr],
                            "type": "rectangle",
                        })
        return results
