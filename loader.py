import os
import numpy as np
import cv2
import OpenEXR
import Imath


def load_hdri(filepath: str) -> np.ndarray:
    """
    Loads an HDR or EXR image into a 32-bit float numpy array (RGB).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.exr':
        return _load_exr(filepath)
    elif ext == '.hdr':
        return _load_hdr(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .hdr or .exr")


def _load_exr(filepath: str) -> np.ndarray:
    """Reads OpenEXR files using the OpenEXR and Imath libraries."""
    exr_file = OpenEXR.InputFile(filepath)

    # Get the image dimensions from the data window
    dw = exr_file.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Define the data type we want to extract (32-bit float)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    # Check if the EXR has standard RGB channels
    header_channels = exr_file.header()['channels'].keys()
    if not all(c in header_channels for c in ['R', 'G', 'B']):
        raise ValueError(f"EXR file {filepath} does not contain standard R, G, B channels.")

    # Read the three color channels
    channels = exr_file.channels(('R', 'G', 'B'), FLOAT)

    # Convert binary buffer to numpy arrays and reshape to image dimensions
    img_data = [np.frombuffer(c, dtype=np.float32).reshape(height, width) for c in channels]

    # Stack into a single (height, width, 3) array
    return np.stack(img_data, axis=-1)


def _load_hdr(filepath: str) -> np.ndarray:
    """Reads Radiance HDR files using OpenCV."""
    # IMREAD_ANYDEPTH ensures we get 32-bit floats instead of 8-bit integers
    img = cv2.imread(filepath, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"OpenCV failed to decode the HDR file: {filepath}")

    # OpenCV loads in BGR order by default, convert to RGB
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)