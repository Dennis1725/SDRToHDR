import os
from typing import Any

from loader import load_hdri
from luminance import analyze_luminance
from statistics import analyze_rgb_energy
from light_sources import detect_light_sources


def analyze_hdri_file(filepath: str) -> tuple[Any, dict]:
    img = load_hdri(filepath)
    lum_stats = analyze_luminance(img)
    energy_stats = analyze_rgb_energy(img)
    light_sources = detect_light_sources(img)

    report_data = {
        "filename": os.path.basename(filepath),
        "path": filepath,
        "resolution": f"{img.shape[1]}x{img.shape[0]}",
        "luminance": lum_stats,
        "energy": energy_stats,
        "light_sources": light_sources,
        "light_source_count": len(light_sources),
        "brightest_light_source_peak": float(max((s["peak_intensity"] for s in light_sources), default=0.0)),
    }

    return img, report_data
