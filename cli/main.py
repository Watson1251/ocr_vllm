import json
import sys
from pathlib import Path

import cv2
import click

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig
from src.models.layout_detector import LayoutDetector
from src.ocr.ocr_engine import OCREngine
from src.pipeline.pdf_loader import pdf_to_images
from src.pipeline.page_processor import process_page_image


@click.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
def main(input_path: str):
    """
    Run YOLO-DocLayNet, save annotated image(s) and JSON with classes + OCR to configured output dir.
    """
    cfg = AppConfig.from_env()
    layout_detector = LayoutDetector(cfg)
    ocr_engine = OCREngine(cfg)

    input_path = Path(input_path)
    out_root = Path(cfg.pipeline.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    doc = {
        "input": str(input_path),
        "num_pages": 0,
        "pages": [],
    }

    if input_path.suffix.lower() == ".pdf":
        for page_num, img_path in pdf_to_images(input_path, dpi=cfg.pipeline.dpi_layout):
            page_result = process_page_image(str(img_path), cfg, layout_detector, ocr_engine)
            page_result["page_num"] = page_num
            doc["pages"].append(
                {
                    "page_num": page_num,
                    "regions": page_result["regions"],
                }
            )
            out_img_path = out_root / f"{input_path.stem}_page_{page_num:04d}_boxes.png"
            cv2.imwrite(str(out_img_path), page_result["image_with_boxes"])
        doc["num_pages"] = len(doc["pages"])
    else:
        page_result = process_page_image(str(input_path), cfg, layout_detector, ocr_engine)
        page_result["page_num"] = 1
        doc["pages"].append(
            {
                "page_num": 1,
                "regions": page_result["regions"],
            }
        )
        out_img_path = out_root / f"{input_path.stem}_boxes.png"
        cv2.imwrite(str(out_img_path), page_result["image_with_boxes"])
        doc["num_pages"] = 1

    out_json_path = out_root / f"{input_path.stem}.json"
    out_json_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    click.echo(f"Saved outputs to {out_root} (images + {out_json_path.name})")


if __name__ == "__main__":
    main()
