from pathlib import Path
from typing import Iterator, Tuple, Union

import fitz  # PyMuPDF


def pdf_to_images(pdf_path: Union[str, Path], dpi: int) -> Iterator[Tuple[int, Path]]:
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        out_dir = pdf_path.with_suffix("")
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"page_{page_idx + 1:04d}.png"
        pix.save(str(out_path))

        yield page_idx + 1, out_path
