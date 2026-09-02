import json
import csv
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def export_json(data: dict, output_dir: str, filename: str = "analysis.json"):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, allow_nan=False)


def export_csv_stats(data: dict, output_dir: str, filename: str = "statistics.csv"):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    flat_data = {k: v for k, v in data.items() if not isinstance(v, list)}
    file_exists = os.path.isfile(filepath)
    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=flat_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(flat_data)


def export_comparison_csv(rows: list[dict], output_dir: str, filename: str = "comparison_statistics.csv"):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fieldnames = ["metric", "image1", "image2", "difference", "ratio"]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_pdf(comparison: dict, output_dir: str, summary_path: str, filename: str = "comparison_report.pdf"):
    """Create a two-page PDF containing the dashboard and a reproducible metric summary."""
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, filename)

    with PdfPages(pdf_path) as pdf:
        try:
            img = plt.imread(summary_path)
            fig1, ax1 = plt.subplots(figsize=(11.7, 8.3))
            ax1.imshow(img)
            ax1.axis('off')
            ax1.set_title('HDR Comparison Summary')
            pdf.savefig(fig1, bbox_inches='tight')
            plt.close(fig1)
        except Exception:
            pass

        fig2, ax2 = plt.subplots(figsize=(11.7, 8.3))
        ax2.axis('off')
        img1 = comparison.get("image1", {})
        img2 = comparison.get("image2", {})
        metrics = comparison.get("metrics", {})
        lum1 = img1.get('luminance', {})
        lum2 = img2.get('luminance', {})
        thresholds = lum1.get('thresholds', {})

        lines = [
            "HDR Comparison",
            f"Reference: {img1.get('filename', '')} ({img1.get('resolution', '')})",
            f"Generated: {img2.get('filename', '')} ({img2.get('resolution', '')})",
            "",
            "HDR Fidelity",
            f"Dynamic Range (EV): ref {lum1.get('dynamic_range_ev', 0.0):.4f}, gen {lum2.get('dynamic_range_ev', 0.0):.4f}",
            f"Robust Dynamic Range (EV): ref {lum1.get('robust_dynamic_range_ev', 0.0):.4f}, gen {lum2.get('robust_dynamic_range_ev', 0.0):.4f}",
            f"Maximum Luminance: ref {lum1.get('max_luminance', 0.0):.6g}, gen {lum2.get('max_luminance', 0.0):.6g}",
            f"Wasserstein (log10 luminance): {metrics.get('wasserstein_distance', {}).get('value', None)}",
            f"SSIM (log10 luminance): {metrics.get('ssim_log_luminance', {}).get('value', None)}",
            "",
            "HDR Region Analysis",
        ]

        for label in ["highlight_pixel_ratio", "shadow_pixel_ratio", "highlight_clipping_ratio", "shadow_clipping_ratio"]:
            m = metrics.get(label, {})
            if m:
                lines.append(f"{label}: ref {m.get('image1', 0.0) * 100:.4f} %, gen {m.get('image2', 0.0) * 100:.4f} %")

        lines += [
            "",
            "Thresholds",
            f"Highlight > {thresholds.get('highlight_threshold', 1000.0)} {thresholds.get('unit', 'EXR units')}",
            f"Shadow < {thresholds.get('shadow_threshold', 0.1)} {thresholds.get('unit', 'EXR units')}",
            f"Highlight clipping > {thresholds.get('highlight_clip_threshold', 10000.0)} {thresholds.get('unit', 'EXR units')}",
            f"Shadow clipping < {thresholds.get('shadow_clip_threshold', 0.01)} {thresholds.get('unit', 'EXR units')}",
            "Absolute cd/m² interpretation requires an absolutely calibrated source EXR.",
            "",
            "ΔE note: ΔE2000 is calculated on normalized/clipped RGB and is supplementary, not a direct HDR luminance metric.",
            "",
            "Outputs:",
        ]
        lines.extend(f"- {out}" for out in comparison.get("outputs", []))

        ax2.text(0.01, 0.99, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9)
        pdf.savefig(fig2, bbox_inches='tight')
        plt.close(fig2)
