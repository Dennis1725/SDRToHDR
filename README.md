# HDR Comparison Toolkit

This project analyzes two HDR images with shared criteria and produces comparison outputs suitable for a research paper.

## What it does

- Loads `.hdr` and `.exr` files
- Computes luminance metrics, RGB energy balance, and detected light sources
- Generates per-image false-color and histogram visuals
- Produces a direct comparison report with plots and summary statistics

## Usage

Analyze one HDR image:

```bash
python main.py path\\to\\image.hdr
```

Compare two HDR images:

```bash
python main.py path\\to\\image1.hdr path\\to\\image2.exr
```

You can also run the comparison entry point directly:

```bash
python compare.py path\\to\\image1.hdr path\\to\\image2.exr
```

## Outputs

- `reports/<image-name>/analysis.json`
- `reports/<image-name>/false_color.png`
- `reports/<image-name>/histogram.png`
- `reports/comparisons/<pair>/comparison.json`
- `reports/comparisons/<pair>/comparison_statistics.csv`
- `reports/comparisons/<pair>/comparison_summary.png`
- `reports/comparisons/<pair>/comparison_summary.pdf`

## Criteria

The main comparison metrics are:

- Dynamic range in EV
- Average luminance
- Maximum luminance
- Median luminance
- RGB energy balance percentages
- Light source count
- Brightest detected light source peak

## Notes

- HDR images are loaded as 32-bit float RGB arrays.
- Equirectangular images use solid-angle weighting so pole regions contribute correctly.
- The comparison figure is designed to work as a paper-ready summary graphic.
