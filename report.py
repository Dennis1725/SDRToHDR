import json
import csv
import os


def export_json(data: dict, output_dir: str, filename: str = "analysis.json"):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def export_csv_stats(data: dict, output_dir: str, filename: str = "statistics.csv"):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    # Flatten dict for CSV writing (excluding nested light sources list)
    flat_data = {k: v for k, v in data.items() if not isinstance(v, list)}

    file_exists = os.path.isfile(filepath)
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=flat_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(flat_data)


def export_comparison_csv(rows: list[dict], output_dir: str, filename: str = "comparison_statistics.csv"):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    file_exists = os.path.isfile(filepath)
    fieldnames = ["metric", "image1", "image2", "difference", "ratio"]
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)