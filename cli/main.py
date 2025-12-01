import json
import sys
import tempfile
from pathlib import Path

import cv2
from flask import Flask, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig
from src.models.layout_detector import LayoutDetector
from src.ocr.ocr_engine import OCREngine
from src.ocr.language_detector import LanguageDetector
from src.pipeline.pdf_loader import pdf_to_images
from src.pipeline.page_processor import process_page_image

cfg = AppConfig.from_env()
layout_detector = LayoutDetector(cfg)
ocr_engine = OCREngine(cfg)
lang_detector = LanguageDetector(default_lang=cfg.ocr.lang)
out_root = Path(cfg.pipeline.output_dir)
out_root.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)


def _run_inference(input_path: Path) -> dict:
    doc: dict = {
        "input": str(input_path),
        "num_pages": 0,
        "pages": [],
    }

    if input_path.suffix.lower() == ".pdf":
        for page_num, img_path in pdf_to_images(input_path, dpi=cfg.pipeline.dpi_layout):
            page_result = process_page_image(str(img_path), cfg, layout_detector, ocr_engine, lang_detector)
            page_result["page_num"] = page_num
            doc["pages"].append(
                {
                    "page_num": page_num,
                    "regions": page_result["regions"],
                }
            )
            out_img_path = out_root / f"{input_path.stem}_page_{page_num:04d}_boxes.png"
            out_img_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_img_path), page_result["image_with_boxes"])
        doc["num_pages"] = len(doc["pages"])
    else:
        page_result = process_page_image(str(input_path), cfg, layout_detector, ocr_engine, lang_detector)
        page_result["page_num"] = 1
        doc["pages"].append(
            {
                "page_num": 1,
                "regions": page_result["regions"],
            }
        )
        out_img_path = out_root / f"{input_path.stem}_boxes.png"
        out_img_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_img_path), page_result["image_with_boxes"])
        doc["num_pages"] = 1

    out_json_path = out_root / f"{input_path.stem}.json"
    out_json_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    doc["output_json"] = str(out_json_path)
    return doc


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/process", methods=["POST"])
def process_endpoint():
    """
    Accepts either:
    - multipart/form-data with file under key 'file'
    - JSON body: {"path": "/absolute/or/relative/path/to/pdf_or_image"}
    Returns JSON with regions and paths to saved outputs.
    """
    temp_file = None
    try:
        file = request.files.get("file")
        json_body = request.get_json(silent=True) or {}
        path_str = json_body.get("path") if isinstance(json_body, dict) else None

        if file:
            suffix = Path(file.filename).suffix or ".png"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            file.save(tmp.name)
            temp_file = tmp
            input_path = Path(tmp.name)
        elif path_str:
            input_path = Path(path_str).expanduser().resolve()
            if not input_path.exists():
                return jsonify({"error": f"path does not exist: {input_path}"}), 400
        else:
            return jsonify({"error": "provide a file upload or JSON with 'path'"}), 400

        result = _run_inference(input_path)
        return jsonify(result), 200
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500
    finally:
        if temp_file:
            Path(temp_file.name).unlink(missing_ok=True)


def _run_startup_example():
    """Run a dev sanity check on startup using data/example.png if present."""
    example_path = PROJECT_ROOT / "data" / "arabic.png"
    if not example_path.exists():
        print(f"[startup] Skipping example; not found: {example_path}")
        return
    try:
        result = _run_inference(example_path)
        print(f"[startup] Example processed: {example_path} -> {result.get('output_json')}")
    except Exception as exc:  # noqa: BLE001
        print(f"[startup] Example failed: {exc}")


_run_startup_example()


if __name__ == "__main__":
    # For local debug (dev server). For production use gunicorn:
    # gunicorn --reload --bind 0.0.0.0:8000 cli.main:app
    app.run(host="0.0.0.0", port=8000, debug=True)
