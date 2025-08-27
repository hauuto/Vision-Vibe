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
    """Calculate histogram equalization for single channel"""
    arr = np.bincount(npArray.flatten(), minlength=256)
    pdf = arr / npArray.size
    cdf = np.cumsum(pdf)
    tf = np.round(cdf * 255)
    img = tf[npArray]
    return img

def histogram_equalization(image):
    """Apply histogram equalization to image"""
    if len(image.shape) == 2:  # Grayscale image
        img_refine = cal_img(image)
        return img_refine.astype(np.uint8)
    
    if len(image.shape) == 3 and image.shape[2] == 3:  # BGR format
        b, g, r = cv2.split(image)
        tf_b = cal_img(b)
        tf_g = cal_img(g)
        tf_r = cal_img(r)
        img_refine = cv2.merge((tf_b, tf_g, tf_r))
        return img_refine.astype(np.uint8)
    
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
