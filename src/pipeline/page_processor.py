from typing import Dict, Any, List

import cv2
import numpy as np

from src.config import AppConfig
from src.models.layout_detector import LayoutDetector, LayoutRegion
from src.utils.image_ops import load_image_bgr

# BGR colors per region type
TYPE_COLORS = {
    "title": (60, 180, 255),
    "subtitle": (120, 160, 255),
    "paragraph": (80, 200, 120),
    "caption": (100, 200, 200),
    "figure": (210, 140, 70),
    "table": (80, 140, 240),
    "page_meta": (200, 200, 80),
    "other": (180, 180, 180),
}


def process_page_image(
    image_path: str,
    cfg: AppConfig,
    layout_detector: LayoutDetector,
    ocr_engine,
) -> Dict[str, Any]:
    img_bgr = load_image_bgr(image_path)

    regions: List[LayoutRegion] = layout_detector.detect(img_bgr)

    # drop tiny boxes that cannot contain text/numbers
    # regions = _filter_too_small(regions, min_area=10.0, min_side=6.0)

    # sort by reading order: top then left
    regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))

    # regions = _suppress_overlaps(regions, iou_threshold=0.5)
    # regions = _resolve_inline_overlaps(regions)
    regions = _filter_empty_boxes(regions, img_bgr, ink_ratio_threshold=0.05)
    # regions = _remove_nested_small(regions, area_ratio_threshold=0.08)
    regions = _expand_boxes(
        regions,
        img_bgr,
        padding=50,
        ink_ratio_threshold=0.01,
    )
    # final guard against tiny boxes introduced by trimming
    # regions = _filter_too_small(regions, min_area=120.0, min_side=8.0)

    detections: List[Dict[str, Any]] = []
    annotated = img_bgr.copy()

    for region in regions:
        x1, y1, x2, y2 = map(int, region.bbox)
        color = TYPE_COLORS.get(region.region_type, (0, 200, 0))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)

        # ocr_res = ocr_engine.ocr_region(img_bgr, region)

        detections.append(
            {
                "bbox": list(map(float, region.bbox)),
                "score": float(region.score),
                "doclaynet_label": region.label,
                "type": region.region_type,
                "text": '' #ocr_res.text,
            }
        )

    return {
        "image_path": image_path,
        "regions": detections,
        "image_with_boxes": annotated,
    }


def _suppress_overlaps(regions: List[LayoutRegion], iou_threshold: float = 0.5) -> List[LayoutRegion]:
    """
    Suppress overlapping/contained boxes: keep the higher-priority/score box when IoU exceeds threshold
    or one box is fully contained within another.
    """
    kept: List[LayoutRegion] = []

    def priority(r: LayoutRegion) -> float:
        # Prefer page headers/footers over text in the same row, then score.
        base = 2.0 if r.region_type == "page_meta" or r.label in {"page-header", "page-footer"} else 1.0
        return base + r.score * 0.1

    def area(r: LayoutRegion) -> float:
        x1, y1, x2, y2 = r.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def iou(a: LayoutRegion, b: LayoutRegion) -> float:
        ax1, ay1, ax2, ay2 = a.bbox
        bx1, by1, bx2, by2 = b.bbox
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0
        union_area = area(a) + area(b) - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def contains(inner: LayoutRegion, outer: LayoutRegion) -> bool:
        ix1, iy1, ix2, iy2 = inner.bbox
        ox1, oy1, ox2, oy2 = outer.bbox
        return ix1 >= ox1 and iy1 >= oy1 and ix2 <= ox2 and iy2 <= oy2

    # iterate regions as-is (already sorted reading order)
    for region in regions:
        drop = False
        r_area = area(region)
        for idx, kept_region in enumerate(list(kept)):
            k_area = area(kept_region)
            overlap = iou(region, kept_region) >= iou_threshold
            contained = contains(region, kept_region) or contains(kept_region, region)

            if overlap or contained:
                # keep higher priority; then larger area; then higher score
                if priority(region) > priority(kept_region) or (
                    priority(region) == priority(kept_region)
                    and ((r_area > k_area) or (r_area == k_area and region.score > kept_region.score))
                ):
                    kept.pop(idx)
                    # continue checking against other kept boxes
                    continue
                else:
                    drop = True
                    break
        if not drop:
            kept.append(region)

    return kept


def _resolve_inline_overlaps(
    regions: List[LayoutRegion],
    vertical_overlap: float = 0.3,
    min_side: float = 8.0,
) -> List[LayoutRegion]:
    """
    For boxes that share a row (sufficient vertical overlap), force them to not overlap horizontally.
    Use confidence/priority to decide which box keeps more width; the other is trimmed to start after it.
    """
    if not regions:
        return regions

    def priority(r: LayoutRegion) -> float:
        base = 2.0 if r.region_type == "page_meta" or r.label in {"page-header", "page-footer"} else 1.0
        return base + r.score * 0.1

    def same_row(a: LayoutRegion, b: LayoutRegion) -> bool:
        ay1, ay2 = a.bbox[1], a.bbox[3]
        by1, by2 = b.bbox[1], b.bbox[3]
        inter = max(0.0, min(ay2, by2) - max(ay1, by1))
        return inter >= vertical_overlap * min(ay2 - ay1, by2 - by1)

    adjusted = regions[:]
    to_remove = set()
    # group by rows
    rows: List[List[LayoutRegion]] = []
    for r in adjusted:
        placed = False
        for row in rows:
            if same_row(row[0], r):
                row.append(r)
                placed = True
                break
        if not placed:
            rows.append([r])

    for row in rows:
        row.sort(key=lambda r: r.bbox[0])
        for i in range(len(row) - 1):
            current = row[i]
            nxt = row[i + 1]
            if nxt.bbox[0] < current.bbox[2]:
                # overlap horizontally; trim the lower-priority one
                if priority(current) >= priority(nxt):
                    new_nx1 = current.bbox[2] + 1.0
                    # avoid inversion
                    if new_nx1 >= nxt.bbox[2]:
                        new_nx1 = min(nxt.bbox[2] - 1.0, new_nx1)
                    if (nxt.bbox[2] - new_nx1) < min_side:
                        to_remove.add(id(nxt))
                    else:
                        nxt.bbox = (new_nx1, nxt.bbox[1], nxt.bbox[2], nxt.bbox[3])
                else:
                    new_cx2 = nxt.bbox[0] - 1.0
                    if new_cx2 <= current.bbox[0]:
                        new_cx2 = max(current.bbox[0] + 1.0, new_cx2)
                    if (new_cx2 - current.bbox[0]) < min_side:
                        to_remove.add(id(current))
                    else:
                        current.bbox = (current.bbox[0], current.bbox[1], new_cx2, current.bbox[3])

    return [r for r in adjusted if id(r) not in to_remove]


def _remove_nested_small(
    regions: List[LayoutRegion],
    area_ratio_threshold: float = 0.08,
) -> List[LayoutRegion]:
    """
    Remove boxes that are fully contained within another and are very small relative to the container.
    """
    kept: List[LayoutRegion] = []
    for i, r in enumerate(regions):
        rx1, ry1, rx2, ry2 = r.bbox
        r_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
        contained = False
        for j, other in enumerate(regions):
            if i == j:
                continue
            ox1, oy1, ox2, oy2 = other.bbox
            if rx1 >= ox1 and ry1 >= oy1 and rx2 <= ox2 and ry2 <= oy2:
                o_area = max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1)
                if o_area > 0 and (r_area / o_area) <= area_ratio_threshold:
                    contained = True
                    break
        if not contained:
            kept.append(r)
    return kept


def _filter_empty_boxes(
    regions: List[LayoutRegion],
    img_bgr: np.ndarray,
    ink_ratio_threshold: float = 0.005,
) -> List[LayoutRegion]:
    """
    Drop boxes whose interior has almost no ink (very light pixels), which are likely false positives.
    """
    if not regions:
        return regions

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    kept: List[LayoutRegion] = []

    for r in regions:
        x1, y1, x2, y2 = map(int, r.bbox)
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = gray[y1:y2, x1:x2]
        dark_ratio = (crop < 240).sum() / float(crop.size)
        if dark_ratio >= ink_ratio_threshold:
            kept.append(r)

    return kept


def _filter_too_small(
    regions: List[LayoutRegion],
    min_area: float = 80.0,
    min_side: float = 6.0,
) -> List[LayoutRegion]:
    """
    Remove boxes that are too small to reasonably contain text/numbers to speed up processing.
    """
    kept: List[LayoutRegion] = []
    for r in regions:
        x1, y1, x2, y2 = r.bbox
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        if w < min_side or h < min_side:
            continue
        if w * h < min_area:
            continue
        kept.append(r)
    return kept


def _expand_boxes(
    regions: List[LayoutRegion],
    img_bgr: np.ndarray,
    padding: int = 32,
    ink_ratio_threshold: float = 0.003,
) -> List[LayoutRegion]:
    """
    First expand each box up to the maximum allowed margin (bounded by neighbors/image),
    then shrink back per-side if the added margin is mostly non-text (very light pixels).
    """
    height, width = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    expanded: List[LayoutRegion] = []

    def overlaps_vert(a: LayoutRegion, b: LayoutRegion) -> bool:
        ay1, ay2 = a.bbox[1], a.bbox[3]
        by1, by2 = b.bbox[1], b.bbox[3]
        return not (ay2 <= by1 or ay1 >= by2)

    def overlaps_horiz(a: LayoutRegion, b: LayoutRegion) -> bool:
        ax1, ax2 = a.bbox[0], a.bbox[2]
        bx1, bx2 = b.bbox[0], b.bbox[2]
        return not (ax2 <= bx1 or ax1 >= bx2)

    def band_ink_ratio(band: np.ndarray) -> float:
        if band.size == 0:
            return 0.0
        dark = (band < 240).sum()
        return dark / band.size

    for i, region in enumerate(regions):
        x1, y1, x2, y2 = region.bbox
        left_allow = min(padding, x1)
        right_allow = min(padding, width - x2)
        up_allow = min(padding, y1)
        down_allow = min(padding, height - y2)

        for j, other in enumerate(regions):
            if i == j:
                continue
            ox1, oy1, ox2, oy2 = other.bbox

            if overlaps_vert(region, other):
                left_allow = min(left_allow, max(0.0, x1 - ox2 - 1))
                right_allow = min(right_allow, max(0.0, ox1 - x2 - 1))

            if overlaps_horiz(region, other):
                up_allow = min(up_allow, max(0.0, y1 - oy2 - 1))
                down_allow = min(down_allow, max(0.0, oy1 - y2 - 1))

        # expand to max allowed first
        max_x1 = max(0.0, x1 - left_allow)
        max_y1 = max(0.0, y1 - up_allow)
        max_x2 = min(float(width - 1), x2 + right_allow)
        max_y2 = min(float(height - 1), y2 + down_allow)

        # measure ink in the added margins
        left_band = gray[int(max_y1) : int(max_y2), int(max_x1) : int(x1)]
        right_band = gray[int(max_y1) : int(max_y2), int(x2) : int(max_x2)]
        top_band = gray[int(max_y1) : int(y1), int(max_x1) : int(max_x2)]
        bottom_band = gray[int(y2) : int(max_y2), int(max_x1) : int(max_x2)]

        shrink_left = 0.0 if band_ink_ratio(left_band) >= ink_ratio_threshold else left_allow
        shrink_right = 0.0 if band_ink_ratio(right_band) >= ink_ratio_threshold else right_allow
        shrink_up = 0.0 if band_ink_ratio(top_band) >= ink_ratio_threshold else up_allow
        shrink_down = 0.0 if band_ink_ratio(bottom_band) >= ink_ratio_threshold else down_allow

        new_x1 = max_x1 + shrink_left
        new_y1 = max_y1 + shrink_up
        new_x2 = max_x2 - shrink_right
        new_y2 = max_y2 - shrink_down

        expanded.append(
            LayoutRegion(
                bbox=(new_x1, new_y1, new_x2, new_y2),
                score=region.score,
                label=region.label,
                region_type=region.region_type,
            )
        )

    return expanded
