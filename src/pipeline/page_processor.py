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

MIN_AREA = 80.0  # min area for boxes to keep
MIN_SIDE = 6.0  # min width/height for boxes to keep


def process_page_image(
    image_path: str,
    cfg: AppConfig,
    layout_detector: LayoutDetector,
    ocr_engine,
    lang_detector,
) -> Dict[str, Any]:
    img_bgr = load_image_bgr(image_path)

    regions: List[LayoutRegion] = layout_detector.detect(img_bgr)

    # drop tiny boxes that cannot contain text/numbers
    regions = _filter_too_small(regions, min_area=MIN_AREA, min_side=MIN_SIDE)

    # # sort by reading order: top then left
    regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))

    regions = _suppress_overlaps(regions, iou_threshold=0.35)
    regions = _resolve_inline_overlaps(regions)
    regions = _filter_empty_boxes(regions, img_bgr, ink_ratio_threshold=0.02)
    regions = _remove_nested_small(regions, area_ratio_threshold=0.08)
    regions = _expand_boxes(
        regions,
        img_bgr,
        padding=300,
        ink_ratio_threshold=0.008,
        shrink_white_threshold=0.99,
    )
    # second pass after expansion to clean up remaining overlaps/empties
    regions = _suppress_overlaps(regions, iou_threshold=0.25)
    regions = _resolve_inline_overlaps(regions)
    regions = _filter_empty_boxes(regions, img_bgr, ink_ratio_threshold=0.02)
    # final guard against tiny boxes introduced by trimming
    regions = _filter_too_small(regions, min_area=MIN_AREA, min_side=MIN_SIDE)

    detections: List[Dict[str, Any]] = []
    annotated = img_bgr.copy()

    for region in regions:
        x1, y1, x2, y2 = map(int, region.bbox)
        color = TYPE_COLORS.get(region.region_type, (0, 200, 0))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
        label = region.label
        score = str(round(region.score, 3))
        cv2.putText(
            annotated,
            label,
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )

        detected_lang, lang_conf = lang_detector.detect(img_bgr[y1:y2, x1:x2])
        # ocr_res = ocr_engine.ocr_region(img_bgr, region, lang=detected_lang)

        detections.append(
            {
                "bbox": list(map(float, region.bbox)),
                "score": float(region.score),
                "doclaynet_label": region.label,
                "type": region.region_type,
                "text": "", #ocr_res.text,
                "lang": detected_lang,
                "lang_conf": lang_conf,
            }
        )

    return {
        "image_path": image_path,
        "regions": detections,
        "image_with_boxes": annotated,
    }


def _suppress_overlaps(regions: List[LayoutRegion], iou_threshold: float = 0.5) -> List[LayoutRegion]:
    """
    Suppress overlapping/contained boxes: keep the higher-confidence/larger box (with header/footer priority)
    when IoU exceeds threshold or one box is fully contained within another.
    """
    kept: List[LayoutRegion] = []

    def priority(r: LayoutRegion) -> float:
        # Prefer page headers/footers over text in the same row, then score.
        base = 2.0 if r.region_type == "page_meta" or r.label in {"page-header", "page-footer"} else 1.0
        return base + r.score

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
                pr = priority(region)
                pk = priority(kept_region)
                better_region = (
                    (pr > pk)
                    or (pr == pk and region.score > kept_region.score)
                    or (pr == pk and region.score == kept_region.score and r_area > k_area)
                )
                if better_region:
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
) -> List[LayoutRegion]:
    """
    For boxes that share a row (sufficient vertical overlap), merge overlapping boxes
    instead of trimming/dropping them. Merge keeps the higher-priority label/score.
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

    resolved: List[LayoutRegion] = []
    for row in rows:
        row.sort(key=lambda r: r.bbox[0])
        merged_row: List[LayoutRegion] = []
        for r in row:
            if not merged_row:
                merged_row.append(r)
                continue
            prev = merged_row[-1]
            # If overlap horizontally, merge to a union box
            if r.bbox[0] < prev.bbox[2]:
                new_x1 = min(prev.bbox[0], r.bbox[0])
                new_y1 = min(prev.bbox[1], r.bbox[1])
                new_x2 = max(prev.bbox[2], r.bbox[2])
                new_y2 = max(prev.bbox[3], r.bbox[3])
                # choose attributes from higher priority
                chosen = prev if priority(prev) >= priority(r) else r
                merged_row[-1] = LayoutRegion(
                    bbox=(new_x1, new_y1, new_x2, new_y2),
                    score=max(prev.score, r.score),
                    label=chosen.label,
                    region_type=chosen.region_type,
                )
            else:
                merged_row.append(r)
        resolved.extend(merged_row)

    return resolved


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
    padding: int = 60,
    ink_ratio_threshold: float = 0.001,
    shrink_white_threshold: float = 0.995,
) -> List[LayoutRegion]:
    """
    Expand boxes outward per-side when the margin contains ink, respecting
    neighbors/image bounds. Then shrink back ONLY inside the added margin,
    never inside the original detection box.
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

    def white_ratio(band: np.ndarray) -> float:
        if band.size == 0:
            return 1.0
        return (band >= 240).sum() / band.size

    for i, region in enumerate(regions):
        # original box
        x1, y1, x2, y2 = region.bbox

        # safety clamp in case detector produced coordinates outside image
        x1 = max(0.0, min(float(width), x1))
        x2 = max(0.0, min(float(width), x2))
        y1 = max(0.0, min(float(height), y1))
        y2 = max(0.0, min(float(height), y2))

        left_allow = min(padding, x1)
        right_allow = min(padding, width - x2)
        up_allow = min(padding, y1)
        down_allow = min(padding, height - y2)

        # respect neighboring boxes
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

        expand_left = expand_right = expand_up = expand_down = 0.0

        # decide which sides to expand based on ink in the margin bands
        if left_allow > 0:
            band = gray[int(y1): int(y2), int(max(0, x1 - left_allow)): int(x1)]
            if band_ink_ratio(band) >= ink_ratio_threshold:
                expand_left = left_allow
        if right_allow > 0:
            band = gray[int(y1): int(y2), int(x2): int(min(width, x2 + right_allow))]
            if band_ink_ratio(band) >= ink_ratio_threshold:
                expand_right = right_allow
        if up_allow > 0:
            band = gray[int(max(0, y1 - up_allow)): int(y1),
                        int(max(0, x1 - expand_left)): int(min(width, x2 + expand_right))]
            if band_ink_ratio(band) >= ink_ratio_threshold:
                expand_up = up_allow
        if down_allow > 0:
            band = gray[int(y2): int(min(height, y2 + down_allow)),
                        int(max(0, x1 - expand_left)): int(min(width, x2 + expand_right))]
            if band_ink_ratio(band) >= ink_ratio_threshold:
                expand_down = down_allow

        # tentative expanded box
        new_x1 = max(0.0, x1 - expand_left)
        new_y1 = max(0.0, y1 - expand_up)
        new_x2 = min(float(width), x2 + expand_right)
        new_y2 = min(float(height), y2 + expand_down)

        # --- shrink ONLY the added margin, not the original box ---

        # left margin shrink
        while (new_x2 - new_x1) > 6 and new_x1 < x1:
            band = gray[int(new_y1): int(new_y2),
                        int(new_x1): int(min(width, new_x1 + 2))]
            if white_ratio(band) >= shrink_white_threshold:
                new_x1 += 1
            else:
                break

        # right margin shrink
        while (new_x2 - new_x1) > 6 and new_x2 > x2:
            band = gray[int(new_y1): int(new_y2),
                        int(max(0, new_x2 - 2)): int(new_x2)]
            if white_ratio(band) >= shrink_white_threshold:
                new_x2 -= 1
            else:
                break

        # top margin shrink
        while (new_y2 - new_y1) > 6 and new_y1 < y1:
            band = gray[int(new_y1): int(min(height, new_y1 + 2)),
                        int(new_x1): int(new_x2)]
            if white_ratio(band) >= shrink_white_threshold:
                new_y1 += 1
            else:
                break

        # bottom margin shrink
        while (new_y2 - new_y1) > 6 and new_y2 > y2:
            band = gray[int(max(0, new_y2 - 2)): int(new_y2),
                        int(new_x1): int(new_x2)]
            if white_ratio(band) >= shrink_white_threshold:
                new_y2 -= 1
            else:
                break

        expanded.append(
            LayoutRegion(
                bbox=(new_x1, new_y1, new_x2, new_y2),
                score=region.score,
                label=region.label,
                region_type=region.region_type,
            )
        )

    return expanded
