from typing import List, Dict, Any
from dataclasses import dataclass

import numpy as np
import cv2
from ultralytics import YOLO

from src.config import AppConfig

DOCLAYNET_CLASSES = [
    "caption",
    "footnote",
    "formula",
    "list-item",
    "page-footer",
    "page-header",
    "picture",
    "section-header",
    "table",
    "text",
    "title",
]


@dataclass
class LayoutRegion:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    score: float
    label: str
    region_type: str


def map_class_to_type(label: str) -> str:
    """Map DocLayNet label → simpler semantic type."""
    if label == "title":
        return "title"
    if label == "section-header":
        return "subtitle"
    if label in {"text", "list-item", "footnote"}:
        return "paragraph"
    if label == "table":
        return "table"
    if label in {"picture", "formula"}:
        return "figure"
    if label in {"page-header", "page-footer"}:
        return "page_meta"
    if label == "caption":
        return "caption"
    return "other"


class LayoutDetector:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.model = YOLO(cfg.layout.model_path)

    def detect(self, image_bgr: np.ndarray) -> List[LayoutRegion]:
        """
        Run YOLO-DocLayNet on a BGR image and return regions.
        """
        # YOLO expects RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        results = self.model(
            image_rgb,
            imgsz=self.cfg.layout.img_size,
            conf=self.cfg.layout.conf_th,
            iou=self.cfg.layout.iou_th,
            device=self.cfg.layout.device,
            verbose=False,
        )

        regions: List[LayoutRegion] = []
        r0 = results[0]

        for box in r0.boxes:
            cls_id = int(box.cls)
            score = float(box.conf)
            if score < self.cfg.pipeline.min_region_score:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label = DOCLAYNET_CLASSES[cls_id]
            rtype = map_class_to_type(label)

            regions.append(
                LayoutRegion(
                    bbox=(x1, y1, x2, y2),
                    score=score,
                    label=label,
                    region_type=rtype,
                )
            )

        return regions
