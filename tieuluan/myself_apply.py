import cv2
import numpy as np
import matplotlib.pyplot as plt

def amBan(image):
    """Negative transformation"""
    return 255 - image

def chuyen_doi_log(image, c=None):
    """Log transformation"""
    if c is None:
        c = 255 / np.log(1 + np.max(image))
    return c * (np.log(1 + image))

def chuyen_doi_mu(image, g=1, c=None):
    """Power transformation (gamma correction)"""
    if c is None:
        c = 255.0 / np.power(255.0, g)
    return c * np.power(image, g)

def cal_img(npArray):
    """Calculate histogram equalization for single channel - FIXED VERSION"""
    npArray = np.clip(npArray, 0, 255).astype(np.uint8)
    arr = np.bincount(npArray.flatten(), minlength=256)
    total_pixels = npArray.size
    pdf = arr.astype(np.float64) / total_pixels
    cdf = np.cumsum(pdf)
    tf = np.round(cdf * 255).astype(np.uint8)
    img = tf[npArray.flatten()].reshape(npArray.shape)
    img_min = np.min(img)
    img_max = np.max(img)
    img_normalized = ((img.astype(np.float64) - img_min) / (img_max - img_min) * 255)
    img = np.round(img_normalized).astype(np.uint8)
    print(f"🔧 Normalized range from [{img_min}, {img_max}] to [0, 255]")
    
    return img.astype(np.uint8)

def histogram_equalization(image):
    """Apply histogram equalization to image - IMPROVED VERSION WITH NORMALIZATION"""
    # Ensure input is uint8
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
        
    if len(image.shape) == 2: 
        result = cal_img(image)
        return result
    
    elif len(image.shape) == 3 and image.shape[2] == 3:  # BGR format
        b, g, r = cv2.split(image)
        tf_b = cal_img(b)
        tf_g = cal_img(g) 
        tf_r = cal_img(r)
        result = cv2.merge((tf_b, tf_g, tf_r))
        return result.astype(np.uint8)
    
    else:
        print(f"Unsupported image format: shape={image.shape}")
        return image

def piecewise_linear_transformation(pixel, r1=100, s1=0, r2=200, s2=255):
    """Piecewise linear transform for a single pixel"""
    if pixel <= r1:
        return (s1 / r1) * pixel if r1 != 0 else 0
    elif pixel <= r2:
        return ((s2 - s1) / (r2 - r1)) * (pixel - r1) + s1
    else:
        return ((255 - s2) / (255 - r2)) * (pixel - r2) + s2 if r2 != 255 else 255

# ==============================
# Các hàm apply (wrapper)
# ==============================
def apply_log_transform(image, c=None):
    """Apply log transformation (color-aware)"""
    if len(image.shape) == 3:  # Color
        channels = cv2.split(image)
        log_channels = []
        for ch in channels:
            result = chuyen_doi_log(ch.astype(np.float32), c)
            result = np.clip(result, 0, 255).astype(np.uint8)
            log_channels.append(result)
        return cv2.merge(log_channels)
    else:  # Grayscale
        result = chuyen_doi_log(image.astype(np.float32), c)
        return np.clip(result, 0, 255).astype(np.uint8)

def apply_power_transform(image, gamma=0.5, c=None):
    """Apply gamma correction (color-aware)"""
    if len(image.shape) == 3:  # Color
        channels = cv2.split(image)
        power_channels = []
        for ch in channels:
            result = chuyen_doi_mu(ch.astype(np.float32), gamma, c)
            result = np.clip(result, 0, 255).astype(np.uint8)
            power_channels.append(result)
        return cv2.merge(power_channels)
    else:  # Grayscale
        result = chuyen_doi_mu(image.astype(np.float32), gamma, c)
        return np.clip(result, 0, 255).astype(np.uint8)

def apply_negative_transform(image):
    """Apply negative transformation"""
    return amBan(image)

def apply_piecewise_linear(image, r1=100, s1=0, r2=200, s2=255):
    """Apply piecewise linear transformation to image"""
    vec_func = np.vectorize(piecewise_linear_transformation)
    result = vec_func(image, r1, s1, r2, s2)
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_CLAHE(image):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to image"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if len(image.shape) == 2:  # Grayscale
        return clahe.apply(image)
    elif len(image.shape) == 3 and image.shape[2] == 3:  # BGR
        b, g, r = cv2.split(image)
        b = clahe.apply(b)
        g = clahe.apply(g)
        r = clahe.apply(r)
        return cv2.merge((b, g, r))
    return image