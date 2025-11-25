from typing import List, Dict, Any


def assemble_document(pages: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Later: handle sections, headings hierarchy, table formatting, etc.
    return {
        "num_pages": len(pages),
        "pages": pages,
    }
