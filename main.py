import os
import sys
import time
from analysis import analyze_hdri_file
from visualization import save_false_color, save_histograms
from report import export_json, export_csv_stats
from compare import compare_hdr_inputs


def analyze_file(filepath: str, output_dir: str):
    print(f"\n--- Analyzing: {os.path.basename(filepath)} ---")
    start_time = time.time()

    # 1. Load and analyze image
    img, report_data = analyze_hdri_file(filepath)
    print(f"Loaded image shape: {img.shape}")

    # 4. Generate Visuals & Reports
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    out_path = os.path.join(output_dir, base_name)
    os.makedirs(out_path, exist_ok=True)

    save_false_color(img, os.path.join(out_path, "false_color.png"))
    save_histograms(img, os.path.join(out_path, "histogram.png"))
    export_json(report_data, out_path)

    # Combine stats for CSV
    flat_stats = {"filename": report_data["filename"]}
    flat_stats.update(report_data["luminance"])
    flat_stats.update(report_data["energy"])
    export_csv_stats(flat_stats, output_dir)

    print(f"Analysis complete in {time.time() - start_time:.2f}s")
    print(f"Dynamic Range: {report_data['luminance']['dynamic_range_ev']:.2f} EV")
    print(f"Detected Highlight Regions: {report_data['highlight_region_count']}")
    print(f"Outputs saved to: {out_path}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_hdr_or_directory> | python main.py <file1> <file2>")
        sys.exit(1)

    out_dir = "reports"

    if len(sys.argv) == 3 and os.path.isfile(sys.argv[1]) and os.path.isfile(sys.argv[2]):
        compare_hdr_inputs(sys.argv[1], sys.argv[2], os.path.join(out_dir, "comparisons"))
    elif len(sys.argv) == 2:
        target = sys.argv[1]
        if os.path.isfile(target):
            analyze_file(target, out_dir)
        elif os.path.isdir(target):
            for f in os.listdir(target):
                if f.lower().endswith(('.hdr', '.exr')):
                    analyze_file(os.path.join(target, f), out_dir)
        else:
            print(f"Invalid path: {target}")
    else:
        print("Usage: python main.py <path_to_hdr_or_directory> | python main.py <file1> <file2>")