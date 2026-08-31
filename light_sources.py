import numpy as np
import cv2
from luminance import calculate_luminance


def detect_light_sources(img: np.ndarray, threshold_pct: float = 99.5) -> list:
    """
    Finds contiguous bright areas (light sources) above a percentile threshold.
    """
    lum = calculate_luminance(img)

    # Calculate threshold based on percentile to adapt to different HDRIs
    threshold_val = np.percentile(lum, threshold_pct)

    # Create binary mask of bright pixels
    mask = np.uint8(lum > threshold_val)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    sources = []
    # Skip label 0 (background)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        # Filter out tiny noise artifacts
        if area < 5:
            continue

        x, y = centroids[i]

        # Extract pixels for this specific light source
        source_mask = (labels == i)
        peak_intensity = np.max(lum[source_mask])
        avg_intensity = np.mean(lum[source_mask])

        sources.append({
            "id": i,
            "x": int(x),
            "y": int(y),
            "area_pixels": int(area),
            "peak_intensity": float(peak_intensity),
            "avg_intensity": float(avg_intensity)
        })

    # Sort by peak intensity descending
    return sorted(sources, key=lambda s: s["peak_intensity"], reverse=True)