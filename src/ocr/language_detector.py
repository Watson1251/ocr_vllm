from typing import List, Optional, Tuple
import numpy as np
import pytesseract
from pytesseract import Output

try:
    from paddleocr import PaddleOCR
except Exception:  # pragma: no cover
    PaddleOCR = None  # type: ignore

# Map basic scripts to Tesseract langs (fallback)
SCRIPT_TO_LANG = {
    "Latin": "eng",
    "Arabic": "ara",
    "Cyrillic": "rus",
    "Devanagari": "hin",
    "Hangul": "kor",
    "Hebrew": "heb",
    "Greek": "ell",
    "Thai": "tha",
    "Katakana": "jpn",
    "Hiragana": "jpn",
    "Han": "chi_sim",
    "Japanese": "jpn",
}


class LanguageDetector:
    """
    Language detector that prefers PaddleOCR recognition confidence across candidate
    languages, with Tesseract OSD as a fallback.

    New:
      - detect_with_confidence(...) -> (lang, confidence, source)
      - detect(...) still returns only lang (for backwards compatibility)
    """

    def __init__(
        self,
        default_lang: str = "eng",
        candidate_langs: Optional[List[str]] = None,
        min_osd_conf: float = 5.0,
    ):
        self.default_lang = default_lang
        self.min_osd_conf = min_osd_conf
        self.candidate_langs = candidate_langs or [
            "eng",
            "ara",
            "rus",
            "jpn",
            "chi_sim",
            "fra",
            "deu",
        ]
        self.paddle_engines = {}

        if PaddleOCR:
            for lang in self.candidate_langs:
                try:
                    # Disable detection, we already have a crop
                    self.paddle_engines[lang] = PaddleOCR(
                        lang=lang,
                        use_angle_cls=False,
                        det=False,
                        rec=True,
                        show_log=False,
                        use_gpu=False,
                    )
                except Exception:
                    continue

    # ---------- backends return (lang, confidence) ----------

    def _detect_with_paddle(
        self, image_bgr: np.ndarray
    ) -> Optional[Tuple[str, float]]:
        """
        Returns (lang, avg_rec_score) where score is in [0, 1].
        """
        if not self.paddle_engines:
            return None

        best_lang: Optional[str] = None
        best_score: float = -1.0

        for lang, engine in self.paddle_engines.items():
            try:
                res = engine.ocr(image_bgr, det=False, rec=True, cls=False)
                # res is list of [ [ [x,y].. ], (text, score) ]
                scores = []
                for line in res:
                    if (
                        len(line) >= 2
                        and isinstance(line[1], (list, tuple))
                        and len(line[1]) >= 2
                    ):
                        scores.append(float(line[1][1]))
                if scores:
                    avg = float(sum(scores) / len(scores))
                    if avg > best_score:
                        best_score = avg
                        best_lang = lang
            except Exception:
                continue

        if best_lang is None:
            return None
        return best_lang, best_score

    def _detect_with_tesseract_osd(
        self, image_bgr: np.ndarray
    ) -> Optional[Tuple[str, float]]:
        """
        Returns (lang, script_conf_norm) where confidence is normalized to [0, 1].
        """
        try:
            osd = pytesseract.image_to_osd(image_bgr, output_type=Output.DICT)
            script = osd.get("script")
            conf_raw = float(osd.get("script_conf", 0.0))
            if script and conf_raw >= self.min_osd_conf:
                lang = SCRIPT_TO_LANG.get(script)
                if lang:
                    # Tesseract script_conf is roughly 0–100 → normalize to 0–1
                    conf_norm = max(0.0, min(1.0, conf_raw / 100.0))
                    return lang, conf_norm
        except Exception:
            pass
        return None

    # ---------- public API ----------

    def detect_with_confidence(
        self, image_bgr: np.ndarray
    ) -> Tuple[str, float, str]:
        """
        Return (lang, confidence, source).

        - source ∈ {"paddle", "tesseract_osd", "default"}
        - confidence is in [0, 1]
        """
        # 1) PaddleOCR
        paddle_res = self._detect_with_paddle(image_bgr)
        if paddle_res is not None:
            lang, conf = paddle_res
            return lang, conf, "paddle"

        # 2) Tesseract OSD
        osd_res = self._detect_with_tesseract_osd(image_bgr)
        if osd_res is not None:
            lang, conf = osd_res
            return lang, conf, "tesseract_osd"

        # 3) Fallback
        return self.default_lang, 0.0, "default"

    def detect(self, image_bgr: np.ndarray) -> str:
        """
        Backwards-compatible wrapper: returns only the language code.
        """
        lang, lang_conf, _ = self.detect_with_confidence(image_bgr)
        return lang, lang_conf
