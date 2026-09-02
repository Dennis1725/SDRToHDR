import json
import sys
import os
import hashlib
import numpy as np

from analysis import analyze_hdri_file
from report import export_json, export_comparison_csv, export_pdf
from visualization import (
    save_comparison_figure,
    compute_wasserstein_distance,
    compute_ssim_index,
    compute_deltaE_map,
    prepare_pixelwise_pair,
)
from luminance import calculate_luminance


def _comparison_metrics(report: dict) -> dict:
    luminance = report["luminance"]
    energy = report["energy"]
    pcts = luminance.get("percentiles", {})
    thresholds = luminance.get("thresholds", {})
    return {
        "dynamic_range_ev": float(luminance["dynamic_range_ev"]),
        "robust_dynamic_range_ev": float(luminance.get("robust_dynamic_range_ev", 0.0)),
        "avg_luminance": float(luminance["avg_luminance"]),
        "max_luminance": float(luminance["max_luminance"]),
        "median_luminance": float(luminance["median_luminance"]),
        "p0_1": float(pcts.get("p0_1", 0.0)),
        "p1": float(pcts.get("p1", 0.0)),
        "p5": float(pcts.get("p5", 0.0)),
        "p50": float(pcts.get("p50", 0.0)),
        "p95": float(pcts.get("p95", 0.0)),
        "p99": float(pcts.get("p99", 0.0)),
        "p99_9": float(pcts.get("p99_9", 0.0)),
        "highlight_pixel_ratio": float(luminance.get("highlight_pixel_ratio", 0.0)),
        "shadow_pixel_ratio": float(luminance.get("shadow_pixel_ratio", 0.0)),
        "highlight_clipping_ratio": float(luminance.get("highlight_clipping_ratio", 0.0)),
        "shadow_clipping_ratio": float(luminance.get("shadow_clipping_ratio", 0.0)),
        "balance_r_pct": float(energy["balance_r_pct"]),
        "balance_g_pct": float(energy["balance_g_pct"]),
        "balance_b_pct": float(energy["balance_b_pct"]),
        "highlight_region_count": int(report.get("highlight_region_count", len(report.get("highlight_regions", [])))),
        "brightest_highlight_region_peak": float(report.get("brightest_highlight_region_peak", 0.0)),
        "highlight_region_threshold": float(report.get("highlight_region_threshold", thresholds.get("highlight_threshold", 0.0))),
    }


def _load_input(path: str):
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        return None, report
    return analyze_hdri_file(path)


def _print_comparison_table(d1: dict, d2: dict):
    metrics1 = _comparison_metrics(d1)
    metrics2 = _comparison_metrics(d2)
    print(f"Comparing:\n1: {d1['filename']}\n2: {d2['filename']}\n")
    print(f"{'Metric':<36} | {'Image 1':<15} | {'Image 2':<15} | {'Diff':<15}")
    print("-" * 98)
    for key in metrics1.keys():
        v1 = metrics1[key]
        v2 = metrics2.get(key, float('nan'))
        try:
            diff = v1 - v2
            print(f"{key:<36} | {v1:<15.4f} | {v2:<15.4f} | {diff:+.4f}")
        except Exception:
            print(f"{key:<36} | {str(v1):<15} | {str(v2):<15} | {'':<15}")


def compare_reports(file1: str, file2: str):
    _, d1 = _load_input(file1)
    _, d2 = _load_input(file2)
    _print_comparison_table(d1, d2)


def compare_hdr_inputs(file1: str, file2: str, output_dir: str = "comparisons"):
    # Raw images are used by analyze_hdri_file and are never resized for HDR statistics.
    img1, report1 = analyze_hdri_file(file1)
    img2, report2 = analyze_hdri_file(file2)

    digest = hashlib.sha1(
        f"{os.path.abspath(file1)}|{os.path.abspath(file2)}".encode("utf-8")
    ).hexdigest()[:8]
    pair_name = (
        f"{os.path.splitext(report1['filename'])[0]}_vs_"
        f"{os.path.splitext(report2['filename'])[0]}_{digest}"
    )
    out_path = os.path.join(output_dir, pair_name)
    os.makedirs(out_path, exist_ok=True)

    comparison = {"image1": report1, "image2": report2, "metrics": {}}
    metrics1 = _comparison_metrics(report1)
    metrics2 = _comparison_metrics(report2)
    csv_rows = []

    for key, value1 in metrics1.items():
        value2 = metrics2.get(key, 0.0)
        ratio = value1 / value2 if value2 != 0 else None
        entry = {
            "image1": value1,
            "image2": value2,
            "difference": value1 - value2,
            "ratio": ratio,
        }
        comparison["metrics"][key] = entry
        csv_rows.append({
            "metric": key,
            "image1": value1,
            "image2": value2,
            "difference": value1 - value2,
            "ratio": ratio if ratio is not None else "",
        })

    # Pixel-wise metrics use a separate comparison pair. Raw HDR statistics remain untouched.
    pix1, pix2 = prepare_pixelwise_pair(img1, img2)
    lum1 = calculate_luminance(pix1)
    lum2 = calculate_luminance(pix2)

    try:
        wdist = compute_wasserstein_distance(lum1, lum2)
        comparison["metrics"]["wasserstein_distance"] = {"value": float(wdist)}
        csv_rows.append({"metric": "wasserstein_distance", "image1": float(wdist), "image2": "", "difference": "", "ratio": ""})
    except Exception as exc:
        comparison["metrics"]["wasserstein_distance"] = {"value": None, "error": str(exc)}

    try:
        ssim_val = compute_ssim_index(lum1, lum2)
        comparison["metrics"]["ssim_log_luminance"] = {"value": float(ssim_val)}
        csv_rows.append({"metric": "ssim_log_luminance", "image1": float(ssim_val), "image2": "", "difference": "", "ratio": ""})
    except Exception as exc:
        comparison["metrics"]["ssim_log_luminance"] = {"value": None, "error": str(exc)}

    try:
        de_map = compute_deltaE_map(pix1, pix2)
        de_finite = de_map[np.isfinite(de_map)]
        de_stats = {
            "mean": float(np.mean(de_finite)),
            "median": float(np.median(de_finite)),
            "p95": float(np.percentile(de_finite, 95.0)),
            "p99": float(np.percentile(de_finite, 99.0)),
            "p99_9": float(np.percentile(de_finite, 99.9)),
            "max": float(np.max(de_finite)),
            "metric_note": "Computed on normalized/clipped RGB; supplementary color metric, not HDR luminance fidelity.",
        }
        comparison["metrics"]["deltaE"] = de_stats
        for stat_name, stat_value in de_stats.items():
            if isinstance(stat_value, (int, float)):
                csv_rows.append({"metric": f"deltaE_{stat_name}", "image1": stat_value, "image2": "", "difference": "", "ratio": ""})
    except Exception as exc:
        de_map = None
        comparison["metrics"]["deltaE"] = {"error": str(exc)}

    summary_path = os.path.join(out_path, "comparison_summary.png")
    save_comparison_figure(img1, report1, img2, report2, summary_path)

    outputs = [
        os.path.basename(summary_path),
        'false_color_ref.png',
        'false_color_gen.png',
        'log_luminance_ref.png',
        'log_luminance_gen.png',
        'luminance_abs_diff.png',
        'luminance_rel_error.png',
        'deltaE_heatmap.png',
        'highlights_ref.png',
        'highlights_gen.png',
        'highlights_diff.png',
        'comparison_statistics.csv',
    ]
    comparison['outputs'] = outputs

    # Export exactly once, after the comparison object is complete.
    export_json(comparison, out_path, filename="comparison.json")
    export_comparison_csv(csv_rows, out_path)
    export_pdf(comparison, out_path, summary_path)

    _print_comparison_table(report1, report2)
    print(f"\nComparison outputs saved to: {out_path}/")
    return comparison


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare.py file1 file2 [output_dir]")
        sys.exit(1)

    file1, file2 = sys.argv[1], sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) >= 4 else "comparisons"

    if file1.lower().endswith('.json') and file2.lower().endswith('.json'):
        compare_reports(file1, file2)
    else:
        compare_hdr_inputs(file1, file2, output_dir)
