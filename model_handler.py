import cv2
import numpy as np
import torch
import torchvision

class ModelHandler:
    def __init__(self, labels):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.labels = labels
        self.load_network(model="yolo_rgb_weights.pt")

    def load_network(self, model):
        try:
            # Загрузка PyTorch модели
            self.model = torch.load(model, map_location=self.device)
            self.model.eval()  # Переключение в режим inference
            self.is_inititated = True
        except Exception as e:
            raise Exception(f"Cannot load model {model}: {e}")

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # Аналогично примеру, подготовка изображения
        shape = im.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, r, (dw, dh)

    def _infer(self, inputs: np.ndarray):
        try:
            # Подготовка изображения
            image = inputs.copy()
            image, ratio, dwdh = self.letterbox(image, auto=False)
            image = image.transpose((2, 0, 1))
            image = np.expand_dims(image, 0)
            image = np.ascontiguousarray(image)

            # Конвертация в tensor
            im_tensor = torch.from_numpy(image).to(self.device)
            im_tensor = im_tensor.float() / 255.0

            # Inference
            with torch.no_grad():
                predictions = self.model(im_tensor)

            # Обработка результатов YOLO
            # (Здесь нужно адаптировать под формат вывода вашей модели)
            if isinstance(predictions, (list, tuple)):
                predictions = predictions[0]  # Берем первый выход если несколько

            predictions = predictions[0]  # Убираем batch dimension
            predictions = predictions.cpu().numpy()

            # Фильтрация по confidence
            conf_mask = predictions[:, 4] > 0.25  # Пример threshold
            predictions = predictions[conf_mask]

            boxes = predictions[:, :4]
            scores = predictions[:, 4]
            class_ids = predictions[:, 5].astype(int)

            # Масштабирование bounding boxes
            boxes[:, [0, 2]] -= dwdh[0] * 2
            boxes[:, [1, 3]] -= dwdh[1] * 2
            boxes[:, :4] /= ratio

            return boxes, class_ids, scores

        except Exception as e:
            print(f"Inference error: {e}")
            return None, None, None

    def infer(self, image, threshold):
        image = np.array(image)
        image = image[:, :, ::-1].copy()  # RGB to BGR
        h, w, _ = image.shape
        
        boxes, class_ids, scores = self._infer(image)

        results = []
        if boxes is not None:
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
