import numpy as np
import cv2
from luminance import calculate_luminance, HIGHLIGHT_THRESHOLD


def detect_light_sources(img: np.ndarray, threshold: float = HIGHLIGHT_THRESHOLD) -> list:
    """Find contiguous bright regions above a fixed luminance threshold."""
    lum = calculate_luminance(img).astype(np.float64, copy=False)
    lum = np.nan_to_num(
        lum,
        nan=0.0,
        neginf=0.0,
        posinf=np.finfo(np.float64).max,
    )
    lum = np.clip(lum, 0.0, None)

    mask = np.uint8(lum > threshold)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    regions = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 5:
            continue

        x, y = centroids[i]
        source_mask = labels == i
        region_values = lum[source_mask]

        regions.append({
            "id": i,
            "x": int(round(x)),
            "y": int(round(y)),
            "area_pixels": area,
            "peak_intensity": float(np.max(region_values)),
            "avg_intensity": float(np.mean(region_values)),
        })

    return sorted(regions, key=lambda s: s["peak_intensity"], reverse=True)
