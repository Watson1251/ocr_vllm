from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class LayoutConfig(BaseModel):
    model_path: str = "weights/yolov10n-doclaynet.pt"
    img_size: int = 1024  # bump for better small-object recall
    conf_th: float = 0.01  # minimum threshold to keep all candidates
    iou_th: float = 0.4
    device: str = "0"  # we keep this CPU for now


class OCRConfig(BaseModel):
    lang: str = "eng"
    # tweak PSM per region type in code
    tesseract_cmd: Optional[str] = None  # set manually if needed


class PipelineConfig(BaseModel):
    dpi_layout: int = 200
    dpi_ocr: int = 300
    min_region_score: float = 0.05  # allow very weak boxes through
    output_dir: str = "outputs"


class AppConfig(BaseModel):
    layout: LayoutConfig = LayoutConfig()
    ocr: OCRConfig = OCRConfig()
    pipeline: PipelineConfig = PipelineConfig()

    @classmethod
    def from_env(cls) -> "AppConfig":
        # later: read from env vars / .env if you want
        return cls()


BASE_DIR = Path(__file__).resolve().parents[1]
