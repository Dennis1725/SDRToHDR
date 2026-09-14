import json
import csv
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def export_json(data: dict, output_dir: str, filename: str = "analysis.json"):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, allow_nan=False)


def export_csv_stats(data: dict, output_dir: str, filename: str = "statistics.csv"):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    flat_data = {k: v for k, v in data.items() if not isinstance(v, list)}

    file_exists = os.path.isfile(filepath)
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=flat_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(flat_data)


def export_comparison_csv(
    rows: list[dict],
    output_dir: str,
    filename: str = "comparison_statistics.csv",
):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    fieldnames = ["metric", "image1", "image2", "difference", "ratio"]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_pdf(
    comparison: dict,
    output_dir: str,
    summary_path: str,
    filename: str = "comparison_report.pdf",
):
    """Create a dashboard page and a reproducible metric-summary page."""
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, filename)

    with PdfPages(pdf_path) as pdf:
        try:
            img = plt.imread(summary_path)
            fig1, ax1 = plt.subplots(figsize=(11.7, 8.3))
            ax1.imshow(img)
            ax1.axis("off")
            ax1.set_title("HDR Comparison Summary")
            pdf.savefig(fig1, bbox_inches="tight")
            plt.close(fig1)
        except Exception:
            pass

        fig2, ax2 = plt.subplots(figsize=(11.7, 8.3))
        ax2.axis("off")

        img1 = comparison.get("image1", {})
        img2 = comparison.get("image2", {})
        metrics = comparison.get("metrics", {})

        lum1 = img1.get("luminance", {})
        lum2 = img2.get("luminance", {})
        thresholds = lum1.get("thresholds", {})

        def metric_value(name, image_key):
            return metrics.get(name, {}).get(image_key, "n/a")

        pair_w = metrics.get("wasserstein_distance", {}).get("value", "n/a")
        pair_ssim = metrics.get("ssim_log_luminance", {}).get("value", "n/a")
        delta = metrics.get("deltaE", {})

        lines = [
            "HDR Comparison",
            f"Reference: {img1.get('filename', '')} ({img1.get('resolution', '')})",
            f"Generated: {img2.get('filename', '')} ({img2.get('resolution', '')})",
            "",
            "HDR Luminance",
            f"Dynamic Range (EV): ref {lum1.get('dynamic_range_ev', 0.0):.4f}, "
            f"gen {lum2.get('dynamic_range_ev', 0.0):.4f}",
            f"Robust Dynamic Range (EV): ref {lum1.get('robust_dynamic_range_ev', 0.0):.4f}, "
            f"gen {lum2.get('robust_dynamic_range_ev', 0.0):.4f}",
            f"Maximum Luminance: ref {lum1.get('max_luminance', 0.0):.6g}, "
            f"gen {lum2.get('max_luminance', 0.0):.6g}",
            f"Median Luminance: ref {lum1.get('median_luminance', 0.0):.6g}, "
            f"gen {lum2.get('median_luminance', 0.0):.6g}",
            "",
            "HDR Region Analysis",
            f"Highlight Ratio: ref {lum1.get('highlight_pixel_ratio', 0.0) * 100:.4f} %, "
            f"gen {lum2.get('highlight_pixel_ratio', 0.0) * 100:.4f} %",
            f"Shadow Ratio: ref {lum1.get('shadow_pixel_ratio', 0.0) * 100:.4f} %, "
            f"gen {lum2.get('shadow_pixel_ratio', 0.0) * 100:.4f} %",
            f"Highlight Clipping: ref {lum1.get('highlight_clipping_ratio', 0.0) * 100:.4f} %, "
            f"gen {lum2.get('highlight_clipping_ratio', 0.0) * 100:.4f} %",
            f"Shadow Clipping: ref {lum1.get('shadow_clipping_ratio', 0.0) * 100:.4f} %, "
            f"gen {lum2.get('shadow_clipping_ratio', 0.0) * 100:.4f} %",
            f"Highlight Pixels: ref {lum1.get('highlight_pixel_count', 0)}, "
            f"gen {lum2.get('highlight_pixel_count', 0)}",
            "",
            "Pairwise Metrics",
            f"SSIM (log10 luminance): {pair_ssim}",
            f"Wasserstein (log10 luminance): {pair_w}",
            f"Mean ΔE2000: {delta.get('mean', 'n/a')}",
            f"Median ΔE2000: {delta.get('median', 'n/a')}",
            f"P95 ΔE2000: {delta.get('p95', 'n/a')}",
            f"P99 ΔE2000: {delta.get('p99', 'n/a')}",
            f"P99.9 ΔE2000: {delta.get('p99_9', 'n/a')}",
            f"Max ΔE2000: {delta.get('max', 'n/a')}",
            "",
            "Methodology",
            "HDR statistics are computed on the original EXR arrays.",
            "Pixelwise metrics are calculated after resizing only when input resolutions differ.",
            "No additional global luminance scaling is applied.",
            "ΔE2000 is supplementary and uses a shared P99.9 RGB normalization.",
            "",
            "Thresholds",
            f"Highlight > {thresholds.get('highlight_threshold', 1000.0)} "
            f"{thresholds.get('unit', 'EXR units')}",
            f"Shadow < {thresholds.get('shadow_threshold', 0.1)} "
            f"{thresholds.get('unit', 'EXR units')}",
            f"Highlight clipping > {thresholds.get('highlight_clip_threshold', 10000.0)} "
            f"{thresholds.get('unit', 'EXR units')}",
            f"Shadow clipping < {thresholds.get('shadow_clip_threshold', 0.01)} "
            f"{thresholds.get('unit', 'EXR units')}",
            "Absolute cd/m² interpretation requires an absolutely calibrated source EXR.",
            "",
            "Interpretation:",
            "SSIM: higher is better.",
            "Wasserstein and ΔE2000: lower is better.",
            "P99/P99.9 are descriptive distribution statistics, not clipping thresholds.",
            "",
            "Outputs:",
        ]

        lines.extend(f"- {out}" for out in comparison.get("outputs", []))

        ax2.text(
            0.01, 0.99,
            "\n".join(lines),
            va="top",
            ha="left",
            family="monospace",
            fontsize=8.5,
        )

        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)
