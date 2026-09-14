import os
from typing import Any

from loader import load_hdri
from luminance import analyze_luminance, HIGHLIGHT_THRESHOLD
from statistics import analyze_rgb_energy
from light_sources import detect_light_sources


def analyze_hdri_file(filepath: str) -> tuple[Any, dict]:
    """Load one HDR image and compute image-level analysis metrics."""
    img = load_hdri(filepath)
    lum_stats = analyze_luminance(img)
    energy_stats = analyze_rgb_energy(img)
    highlight_regions = detect_light_sources(img, threshold=HIGHLIGHT_THRESHOLD)

    report_data = {
        "filename": os.path.basename(filepath),
        "path": filepath,
        "resolution": f"{img.shape[1]}x{img.shape[0]}",
        "luminance": lum_stats,
        "energy": energy_stats,
        "highlight_regions": highlight_regions,
        "highlight_region_count": len(highlight_regions),
        "brightest_highlight_region_peak": float(
            max((s["peak_intensity"] for s in highlight_regions), default=0.0)
        ),
        "highlight_region_threshold": float(HIGHLIGHT_THRESHOLD),
    }
    return img, report_data
