import numpy as np
from ultralytics import YOLO

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    _SAHI_AVAILABLE = True
except Exception:
    _SAHI_AVAILABLE = False


class ModelHandler:
    def __init__(
        self,
        labels: dict,
        use_sahi: bool = True,
        tile: int = 512,
        overlap: float = 0.20,
        device: str = "cpu",
        postprocess_type: str = "NMS",
        weights_path: str = "yolo_rgb_weights.pt",
        ):

        self.device = device
        self.labels = labels

        self.use_sahi = use_sahi and _SAHI_AVAILABLE
        self.tile = int(tile)
        self.overlap = float(overlap)
        self.postprocess_type = postprocess_type

        self._load_network(weights_path)


    def _load_network(self, weights_path: str):
        try:
            self.model = YOLO(weights_path)
            self.model.to(self.device)
            self.is_inititated = True
        except Exception as e:
            raise RuntimeError(f"Cannot load YOLO model: {e}")

        self.sahi_model = None
        if self.use_sahi:
            try:
                self.sahi_model = AutoDetectionModel.from_pretrained(
                    model_type="ultralytics",
                    model_path=weights_path,
                    device=self.device,
                    confidence_threshold=0.10,
                )
            except Exception as e:
                print(f"[WARN] SAHI init failed ({e}).")
                self.sahi_model = None


    def infer(self, image_pil, threshold: float):

        th = float(threshold)
        if self.sahi_model is not None:
            try:
                self.sahi_model.confidence_threshold = th

                sahi_res = get_sliced_prediction(
                    image=image_pil,
                    detection_model=self.sahi_model,
                    slice_height=self.tile,
                    slice_width=self.tile,
                    overlap_height_ratio=self.overlap,
                    overlap_width_ratio=self.overlap,
                    postprocess_type=self.postprocess_type,
                    postprocess_match_metric="IOU",
                    postprocess_match_threshold=0.50,
                )

                out = []
                for p in sahi_res.object_prediction_list:
                    x1, y1, x2, y2 = p.bbox.to_xyxy()
                    score = float(getattr(p.score, "value", 0.0))

                    class_id = p.category.id
                    class_name = p.category.name if p.category else "unknown"
                    try:
                        class_id_int = int(class_id)
                    except Exception:
                        class_id_int = None

                    if score < th:
                        continue

                    out.append({
                        "confidence": f"{score}",
                        "label": self.labels.get(class_id_int, class_name or "unknown"),
                        "points": [int(x1), int(y1), int(x2), int(y2)],
                        "type": "rectangle",
                    })
                return out
            except Exception as e:
                print(f"[WARN] SAHI inference failed ({e}). Fallback to whole-image.")

        return self._infer(image_pil, th)


    def _infer(self, image_pil, threshold: float):

        try:
            results = self.model(image_pil, verbose=False)
            if not results:
                return []

            result = results[0]
            if getattr(result, "boxes", None) is None or len(result.boxes) == 0:
                return []

            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            h, w = (np.array(image_pil).shape[:2])

            out = []
            for box, cls_id, score in zip(boxes, classes, scores):
                if float(score) < threshold:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box]
                x1 = max(0, min(x1, w))
                x2 = max(0, min(x2, w))
                y1 = max(0, min(y1, h))
                y2 = max(0, min(y2, h))
                if x2 > x1 and y2 > y1:
                    out.append({
                        "confidence": f"{float(score)}",
                        "label": self.labels.get(int(cls_id), "unknown"),
                        "points": [x1, y1, x2, y2],
                        "type": "rectangle",
                    })
            return out

        except Exception as e:
            print(f"[ERROR] Whole-image inference failed: {e}")
            return []

