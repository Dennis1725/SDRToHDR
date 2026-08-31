import numpy as np
import cv2
import matplotlib.pyplot as plt
from luminance import calculate_luminance
import os


def create_false_color_image(img: np.ndarray) -> np.ndarray:
    """Maps log-luminance to a jet colormap for visual brightness inspection."""
    lum = calculate_luminance(img)

    lum_log = np.log(lum + 1e-6)
    lum_norm = cv2.normalize(lum_log, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    lum_8bit = np.uint8(lum_norm)
    return cv2.applyColorMap(lum_8bit, cv2.COLORMAP_JET)


def save_false_color(img: np.ndarray, output_path: str):
    false_color = create_false_color_image(img)
    cv2.imwrite(output_path, false_color)


def save_histograms(img: np.ndarray, output_path: str):
    """Generates a logarithmic histogram of the RGB channels."""
    r = np.log10(np.clip(img[..., 0].ravel(), 1e-6, None))
    g = np.log10(np.clip(img[..., 1].ravel(), 1e-6, None))
    b = np.log10(np.clip(img[..., 2].ravel(), 1e-6, None))

    plt.figure(figsize=(10, 6))
    plt.hist(r, bins=100, color='red', alpha=0.5, label='Red (log10)')
    plt.hist(g, bins=100, color='green', alpha=0.5, label='Green (log10)')
    plt.hist(b, bins=100, color='blue', alpha=0.5, label='Blue (log10)')

    plt.title('RGB Energy Distribution (Logarithmic)')
    plt.xlabel('Log10(Intensity)')
    plt.ylabel('Pixel Count')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(output_path)
    plt.close()


def save_comparison_figure(img1: np.ndarray, report1: dict, img2: np.ndarray, report2: dict, output_path: str):
    """Creates a paper-friendly comparison figure for two HDR images."""
    false_color_1 = cv2.cvtColor(create_false_color_image(img1), cv2.COLOR_BGR2RGB)
    false_color_2 = cv2.cvtColor(create_false_color_image(img2), cv2.COLOR_BGR2RGB)

    lum1 = calculate_luminance(img1)
    lum2 = calculate_luminance(img2)
    log_lum1 = np.log10(np.clip(lum1.ravel(), 1e-6, None))
    log_lum2 = np.log10(np.clip(lum2.ravel(), 1e-6, None))

    luminance_metrics = [
        ("dynamic_range_ev", "Dynamic range (EV)"),
        ("avg_luminance", "Average luminance"),
        ("max_luminance", "Max luminance"),
        ("median_luminance", "Median luminance"),
    ]
    balance_metrics = [
        ("balance_r_pct", "Red %"),
        ("balance_g_pct", "Green %"),
        ("balance_b_pct", "Blue %"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    axes[0, 0].imshow(false_color_1)
    axes[0, 0].set_title(report1["filename"])
    axes[0, 0].axis("off")

    axes[0, 1].imshow(false_color_2)
    axes[0, 1].set_title(report2["filename"])
    axes[0, 1].axis("off")

    x = np.arange(len(luminance_metrics))
    width = 0.35
    values_1 = [report1["luminance"][key] for key, _ in luminance_metrics]
    values_2 = [report2["luminance"][key] for key, _ in luminance_metrics]
    axes[0, 2].bar(x - width / 2, values_1, width, label=report1["filename"])
    axes[0, 2].bar(x + width / 2, values_2, width, label=report2["filename"])
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels([label for _, label in luminance_metrics], rotation=20, ha="right")
    axes[0, 2].set_title("Luminance criteria")
    axes[0, 2].set_ylabel("Value")
    axes[0, 2].grid(True, axis="y", alpha=0.3)
    axes[0, 2].legend(fontsize=8)

    x = np.arange(len(balance_metrics))
    balance_1 = [report1["energy"][key] for key, _ in balance_metrics]
    balance_2 = [report2["energy"][key] for key, _ in balance_metrics]
    axes[1, 0].bar(x - width / 2, balance_1, width, label=report1["filename"])
    axes[1, 0].bar(x + width / 2, balance_2, width, label=report2["filename"])
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([label for _, label in balance_metrics])
    axes[1, 0].set_title("RGB energy balance")
    axes[1, 0].set_ylabel("Percent")
    axes[1, 0].grid(True, axis="y", alpha=0.3)
    axes[1, 0].legend(fontsize=8)

    summary = "\n".join([
        f"{report1['filename']}:",
        f"  Light sources: {report1.get('light_source_count', 0)}",
        f"  Brightest peak: {report1.get('brightest_light_source_peak', 0.0):.3f}",
        f"  Dynamic range: {report1['luminance']['dynamic_range_ev']:.2f} EV",
        "",
        f"{report2['filename']}:",
        f"  Light sources: {report2.get('light_source_count', 0)}",
        f"  Brightest peak: {report2.get('brightest_light_source_peak', 0.0):.3f}",
        f"  Dynamic range: {report2['luminance']['dynamic_range_ev']:.2f} EV",
    ])
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.02,
        0.98,
        summary,
        va="top",
        ha="left",
        family="monospace",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    axes[1, 2].hist(log_lum1, bins=80, alpha=0.55, label=report1["filename"])
    axes[1, 2].hist(log_lum2, bins=80, alpha=0.55, label=report2["filename"])
    axes[1, 2].set_title("Log luminance distribution")
    axes[1, 2].set_xlabel("log10(luminance)")
    axes[1, 2].set_ylabel("Pixel count")
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend(fontsize=8)

    fig.suptitle("HDR comparison", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    root, _ = os.path.splitext(output_path)
    fig.savefig(f"{root}.pdf", bbox_inches="tight")
    plt.close(fig)