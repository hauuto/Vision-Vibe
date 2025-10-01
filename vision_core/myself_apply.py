import cv2
import numpy as np
from typing import Optional, Tuple, Union
import warnings

# Suppress OpenCV warnings
warnings.filterwarnings('ignore', category=UserWarning)


# ==============================
# Core Transformation Functions
# ==============================

def amBan(image: np.ndarray) -> np.ndarray:
    """Negative transformation - optimized"""
    return 255 - image

def chuyen_doi_log(image: np.ndarray, c: Optional[float] = None) -> np.ndarray:
    """Log transformation - optimized"""
    if c is None:
        c = 255 / np.log(1 + np.max(image))

    # Ensure non-negative values and avoid log(0)
    safe_image = np.maximum(image, 1e-10)
    return c * np.log(1 + safe_image)

def chuyen_doi_mu(image: np.ndarray, g: float = 1, c: Optional[float] = None) -> np.ndarray:
    """Power transformation (gamma correction) - optimized"""
    if c is None:
        c = 255.0 / np.power(255.0, g)

    # Normalize to [0,1] range for gamma correction
    normalized = image.astype(np.float32) / 255.0
    corrected = c * np.power(normalized, g)
    return np.clip(corrected * 255, 0, 255)

# ==============================
# Histogram Processing
# ==============================

def cal_img(npArray: np.ndarray) -> np.ndarray:
    """Optimized histogram equalization for single channel"""
    # Ensure uint8 input
    if npArray.dtype != np.uint8:
        npArray = np.clip(npArray, 0, 255).astype(np.uint8)

    # Calculate histogram
    hist = cv2.calcHist([npArray], [0], None, [256], [0, 256]).flatten()

    # Calculate CDF
    cdf = np.cumsum(hist)
    cdf_normalized = cdf * 255 / cdf[-1]

    # Apply transformation
    result = cdf_normalized[npArray]
    return np.clip(result, 0, 255).astype(np.uint8)

def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Improved histogram equalization with better handling"""
    image = np.clip(image, 0, 255).astype(np.uint8)

    if len(image.shape) == 2:  # Grayscale
        return cal_img(image)

    elif len(image.shape) == 3 and image.shape[2] == 3:  # Color (BGR)
        # Process each channel separately
        channels = cv2.split(image)
        equalized_channels = [cal_img(ch) for ch in channels]
        return cv2.merge(equalized_channels)

    else:
        raise ValueError(f"Unsupported image format: shape={image.shape}")

# ==============================
# Apply Functions (Optimized)
# ==============================

def apply_log_transform(image: np.ndarray, c: Optional[float] = None) -> np.ndarray:
    """Apply log transformation with proper type handling"""
    def process_channel(channel):
        result = chuyen_doi_log(channel.astype(np.float32), c)
        return np.clip(result, 0, 255).astype(np.uint8)

    if len(image.shape) == 3:
        channels = cv2.split(image)
        processed_channels = [process_channel(ch) for ch in channels]
        return cv2.merge(processed_channels)
    else:
        return process_channel(image)

def apply_power_transform(image: np.ndarray, gamma: float = 0.5, c: Optional[float] = None) -> np.ndarray:
    """Apply gamma correction with proper type handling"""
    def process_channel(channel):
        result = chuyen_doi_mu(channel.astype(np.float32), gamma, c)
        return np.clip(result, 0, 255).astype(np.uint8)

    if len(image.shape) == 3:
        channels = cv2.split(image)
        processed_channels = [process_channel(ch) for ch in channels]
        return cv2.merge(processed_channels)
    else:
        return process_channel(image)

def apply_negative_transform(image: np.ndarray) -> np.ndarray:
    """Apply negative transformation"""
    return amBan(image)

def apply_piecewise_linear(image: np.ndarray, r1: int = 100, s1: int = 0,
                          r2: int = 200, s2: int = 255) -> np.ndarray:
    """Optimized piecewise linear transformation"""
    def piecewise_transform(pixel_values):
        result = np.zeros_like(pixel_values, dtype=np.float32)

        # Create masks for different ranges
        mask1 = pixel_values <= r1
        mask2 = (pixel_values > r1) & (pixel_values <= r2)
        mask3 = pixel_values > r2

        # Apply transformations
        if r1 != 0:
            result[mask1] = (s1 / r1) * pixel_values[mask1]

        if r2 != r1:
            result[mask2] = ((s2 - s1) / (r2 - r1)) * (pixel_values[mask2] - r1) + s1

        if r2 != 255:
            result[mask3] = ((255 - s2) / (255 - r2)) * (pixel_values[mask3] - r2) + s2
        else:
            result[mask3] = 255

        return result

    transformed = piecewise_transform(image.astype(np.float32))
    return np.clip(transformed, 0, 255).astype(np.uint8)

def apply_CLAHE(image: np.ndarray, clip_limit: float = 2.0,
               tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply CLAHE with configurable parameters"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(image.shape) == 2:  # Grayscale
        return clahe.apply(image)
    elif len(image.shape) == 3 and image.shape[2] == 3:  # Color
        # Convert to LAB color space for better results
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])  # Apply to L channel only
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return image
