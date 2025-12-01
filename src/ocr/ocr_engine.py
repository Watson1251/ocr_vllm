from typing import Optional
from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract

from src.config import AppConfig
from src.models.layout_detector import LayoutRegion


@dataclass
class OCRResult:
    text: str
    region: LayoutRegion
    lang: Optional[str] = None


class OCREngine:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        if cfg.ocr.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = cfg.ocr.tesseract_cmd

    def _psm_for_type(self, region_type: str) -> str:
        if region_type in {"title", "subtitle"}:
            return "7"  # single line
        if region_type in {"paragraph", "caption"}:
            return "6"  # block
        if region_type == "table":
            return "6"
        # fallback
        return "6"

    def ocr_region(
        self,
        image_bgr: np.ndarray,
        region: LayoutRegion,
        lang: Optional[str] = None,
    ) -> OCRResult:
        x1, y1, x2, y2 = map(int, region.bbox)
        crop = image_bgr[y1:y2, x1:x2]

        psm = self._psm_for_type(region.region_type)
        config = f"--psm {psm}"

        text = pytesseract.image_to_string(
            crop,
            lang=lang or self.cfg.ocr.lang,
            config=config,
        )

        return OCRResult(text=text.strip(), region=region, lang=lang or self.cfg.ocr.lang)
