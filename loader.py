import os
import numpy as np
import cv2
import OpenEXR
import Imath


def load_hdri(filepath: str) -> np.ndarray:
    """
    Load HDR/EXR into float32 RGB without global luminance normalization.

    EXR channel values are preserved numerically. Therefore the evaluation
    reports them as EXR units and does not claim cd/m² without absolute
    calibration of the source.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".exr":
        return _load_exr(filepath)
    if ext == ".hdr":
        return _load_hdr(filepath)

    raise ValueError(f"Unsupported file format: {ext}. Use .hdr or .exr")


def _load_exr(filepath: str) -> np.ndarray:
    """Read R/G/B EXR channels as raw float32 values."""
    exr_file = OpenEXR.InputFile(filepath)

    try:
        dw = exr_file.header()["dataWindow"]
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        header_channels = exr_file.header()["channels"].keys()

        if not all(c in header_channels for c in ["R", "G", "B"]):
            raise ValueError(
                f"EXR file {filepath} does not contain standard R, G, B channels."
            )

        channels = exr_file.channels(("R", "G", "B"), FLOAT)
        img_data = [
            np.frombuffer(c, dtype=np.float32).reshape(height, width)
            for c in channels
        ]
        return np.stack(img_data, axis=-1)
    finally:
        exr_file.close()


def _load_hdr(filepath: str) -> np.ndarray:
    """Read Radiance HDR through OpenCV and convert BGR to RGB."""
    img = cv2.imread(
        filepath,
        cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR,
    )

    if img is None:
        raise ValueError(f"OpenCV failed to decode the HDR file: {filepath}")

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
