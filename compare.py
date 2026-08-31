import json
import sys
import os
import hashlib

from analysis import analyze_hdri_file
from report import export_json, export_comparison_csv
from visualization import save_comparison_figure


def _comparison_metrics(report: dict) -> dict:
    luminance = report["luminance"]
    energy = report["energy"]
    return {
        "dynamic_range_ev": float(luminance["dynamic_range_ev"]),
        "avg_luminance": float(luminance["avg_luminance"]),
        "max_luminance": float(luminance["max_luminance"]),
        "median_luminance": float(luminance["median_luminance"]),
        "balance_r_pct": float(energy["balance_r_pct"]),
        "balance_g_pct": float(energy["balance_g_pct"]),
        "balance_b_pct": float(energy["balance_b_pct"]),
        "light_source_count": int(report.get("light_source_count", len(report.get("light_sources", [])))),
        "brightest_light_source_peak": float(report.get("brightest_light_source_peak", 0.0)),
    }


def _load_input(path: str) -> tuple[object, dict]:
    if path.lower().endswith(".json"):
        with open(path, "r") as handle:
            report = json.load(handle)
        return None, report
    return analyze_hdri_file(path)


def _print_comparison_table(d1: dict, d2: dict):
    metrics1 = _comparison_metrics(d1)
    metrics2 = _comparison_metrics(d2)
    print(f"Comparing:\n1: {d1['filename']}\n2: {d2['filename']}\n")
    print(f"{'Metric':<24} | {'Image 1':<15} | {'Image 2':<15} | {'Diff'}")
    print("-" * 72)

    for key in [
        "dynamic_range_ev",
        "avg_luminance",
        "max_luminance",
        "median_luminance",
        "balance_r_pct",
        "balance_g_pct",
        "balance_b_pct",
        "light_source_count",
        "brightest_light_source_peak",
    ]:
        v1 = metrics1[key]
        v2 = metrics2[key]
        diff = v1 - v2
        print(f"{key:<24} | {v1:<15.4f} | {v2:<15.4f} | {diff:+.4f}")


def compare_reports(file1: str, file2: str):
    _, d1 = _load_input(file1)
    _, d2 = _load_input(file2)
    _print_comparison_table(d1, d2)


def compare_hdr_inputs(file1: str, file2: str, output_dir: str):
    img1, report1 = analyze_hdri_file(file1)
    img2, report2 = analyze_hdri_file(file2)

    digest = hashlib.sha1(f"{os.path.abspath(file1)}|{os.path.abspath(file2)}".encode("utf-8")).hexdigest()[:8]
    pair_name = f"{os.path.splitext(report1['filename'])[0]}_vs_{os.path.splitext(report2['filename'])[0]}_{digest}"
    out_path = os.path.join(output_dir, pair_name)
    os.makedirs(out_path, exist_ok=True)

    comparison = {
        "image1": report1,
        "image2": report2,
        "metrics": {},
    }

    metrics1 = _comparison_metrics(report1)
    metrics2 = _comparison_metrics(report2)
    csv_rows = []
    for key, value1 in metrics1.items():
        value2 = metrics2[key]
        ratio = value1 / value2 if value2 not in (0, 0.0) else None
        comparison["metrics"][key] = {
            "image1": value1,
            "image2": value2,
            "difference": value1 - value2,
            "ratio": ratio,
        }
        csv_rows.append({
            "metric": key,
            "image1": value1,
            "image2": value2,
            "difference": value1 - value2,
            "ratio": ratio if ratio is not None else "",
        })

    export_json(comparison, out_path, filename="comparison.json")
    export_comparison_csv(csv_rows, out_path)
    save_comparison_figure(img1, report1, img2, report2, os.path.join(out_path, "comparison_summary.png"))

    _print_comparison_table(report1, report2)
    print(f"\nComparison outputs saved to: {out_path}/")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare.py report1.json report2.json | python compare.py file1 file2")
    else:
        compare_reports(sys.argv[1], sys.argv[2])