import cv2
import numpy as np
from typing import Callable, Tuple, Dict, Any

# Import point processing functions
try:
    from .myself_apply import (
        apply_log_transform,
        apply_power_transform,
        apply_negative_transform,
        histogram_equalization,
        apply_piecewise_linear,
        apply_CLAHE as custom_CLAHE,
    )
except Exception:
    # Fallback if relative import fails in some execution contexts
    from vision_core.myself_apply import (
        apply_log_transform,
        apply_power_transform,
        apply_negative_transform,
        histogram_equalization,
        apply_piecewise_linear,
        apply_CLAHE as custom_CLAHE,
    )


def processing_image(img: np.ndarray, func: Callable[[np.ndarray], np.ndarray], **kwargs) -> np.ndarray:
    """
    Apply a single-channel processing function to either a grayscale image (2D) or
    to each channel of a color image (3-channel or 4-channel keeping alpha).
    If img has shape (H, W), apply directly.
    If img has shape (H, W, 3), split into B,G,R, apply func to each, then merge.
    If img has shape (H, W, 4), process first 3 channels and keep alpha.
    """
    if img is None:
        raise ValueError("processing_image: input is None")

    if img.ndim == 2:
        return func(img, **kwargs)

    if img.ndim == 3:
        h, w, c = img.shape
        if c == 1:
            return func(img[:, :, 0], **kwargs)
        elif c == 3:
            b, g, r = cv2.split(img)
            b = func(b, **kwargs)
            g = func(g, **kwargs)
            r = func(r, **kwargs)
            return cv2.merge([b, g, r])
        elif c == 4:
            b, g, r, a = cv2.split(img)
            b = func(b, **kwargs)
            g = func(g, **kwargs)
            r = func(r, **kwargs)
            return cv2.merge([b, g, r, a])

    raise ValueError(f"Unsupported image shape: {img.shape}")


# -----------------------------
# Convolution utilities
# -----------------------------

def _convolve2d(src: np.ndarray, kernel: np.ndarray, padding: str = 'reflect') -> np.ndarray:
    """Simple 2D convolution for single channel images using given padding.
    Performs correlation (same orientation as kernel)."""
    if src.ndim != 2:
        raise ValueError("_convolve2d expects 2D array")

    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    if padding == 'reflect':
        padded = np.pad(src, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    elif padding == 'edge':
        padded = np.pad(src, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    else:
        padded = np.pad(src, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    kernel = kernel.astype(np.float32)
    src_f = padded.astype(np.float32)
    out = np.zeros_like(src_f, dtype=np.float32)

    H, W = src.shape
    for i in range(H):
        for j in range(W):
            region = src_f[i:i + kh, j:j + kw]
            out[i + pad_h, j + pad_w] = np.sum(region * kernel)

    return out[pad_h:-pad_h or None, pad_w:-pad_w or None]


def _gaussian_kernel(ksize: int, sigma: float) -> np.ndarray:
    ax = np.arange(-(ksize // 2), ksize // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


# -----------------------------
# Noise generators (NumPy/OpenCV)
# -----------------------------

def add_gaussian_noise(img: np.ndarray, mean: float = 0.0, var: float = 10.0) -> np.ndarray:
    """Add Gaussian noise with given mean and variance (on 0-255 scale)."""
    def _fn(ch):
        noise = np.random.normal(mean, np.sqrt(var), ch.shape)
        noisy = ch.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    return processing_image(img, _fn)


def add_salt_pepper_noise(img: np.ndarray, amount: float = 0.02, s_vs_p: float = 0.5) -> np.ndarray:
    """Add salt & pepper noise. amount is proportion of pixels to corrupt."""
    rng = np.random.default_rng()

    def _fn(ch):
        out = ch.copy()
        num_pixels = ch.size
        num_salt = int(amount * num_pixels * s_vs_p)
        num_pepper = int(amount * num_pixels * (1.0 - s_vs_p))

        coords = (rng.integers(0, ch.shape[0], num_salt), rng.integers(0, ch.shape[1], num_salt))
        out[coords] = 255
        coords = (rng.integers(0, ch.shape[0], num_pepper), rng.integers(0, ch.shape[1], num_pepper))
        out[coords] = 0
        return out

    return processing_image(img, _fn)


# -----------------------------
# Blurs / Smoothing (custom)
# -----------------------------

def mean_blur(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    ksize = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    kernel = np.ones((ksize, ksize), dtype=np.float32) / (ksize * ksize)

    def _fn(ch):
        res = _convolve2d(ch, kernel, padding='reflect')
        return np.clip(res, 0, 255).astype(np.uint8)

    return processing_image(img, _fn)


def gaussian_blur(img: np.ndarray, ksize: int = 3, sigma: float = 1.0) -> np.ndarray:
    ksize = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    sigma = max(0.1, float(sigma))
    kernel = _gaussian_kernel(ksize, sigma)

    def _fn(ch):
        res = _convolve2d(ch, kernel, padding='reflect')
        return np.clip(res, 0, 255).astype(np.uint8)

    return processing_image(img, _fn)


def median_blur(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    ksize = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1

    def _fn(ch):
        return cv2.medianBlur(ch, ksize)

    return processing_image(img, _fn)


def bilateral_blur(img: np.ndarray, diameter: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    diameter = int(diameter) if int(diameter) > 0 else 9
    sigma_color = float(sigma_color)
    sigma_space = float(sigma_space)

    def _fn(ch):
        return cv2.bilateralFilter(ch, diameter, sigma_color, sigma_space)

    return processing_image(img, _fn)


# -----------------------------
# Sharpening (custom)
# -----------------------------

def laplacian_sharpen(img: np.ndarray, alpha: float = 1.0, kernel_type: str = '4') -> np.ndarray:
    """Sharpen by subtracting Laplacian (from scratch). kernel_type '4' or '8' connectivity."""
    if kernel_type == '8':
        kernel = np.array([[1, 1, 1],
                           [1, -8, 1],
                           [1, 1, 1]], dtype=np.float32)
    else:
        kernel = np.array([[0, -1, 0],
                           [-1, 4, -1],
                           [0, -1, 0]], dtype=np.float32)

    alpha = float(alpha)

    def _fn(ch):
        lap = _convolve2d(ch, kernel, padding='reflect')
        res = ch.astype(np.float32) - alpha * lap
        return np.clip(res, 0, 255).astype(np.uint8)

    return processing_image(img, _fn)


def unsharp_mask(img: np.ndarray, ksize: int = 5, sigma: float = 1.0, amount: float = 1.0, threshold: int = 0) -> np.ndarray:
    ksize = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    sigma = max(0.1, float(sigma))
    amount = float(amount)
    threshold = max(0, int(threshold))

    def _fn(ch):
        blur = gaussian_blur(ch, ksize, sigma)
        mask = ch.astype(np.int16) - blur.astype(np.int16)
        if threshold > 0:
            low_contrast_mask = np.abs(mask) < threshold
            mask[low_contrast_mask] = 0
        res = ch.astype(np.int16) + (amount * mask)
        return np.clip(res, 0, 255).astype(np.uint8)

    return processing_image(img, _fn)


# -----------------------------
# Edge Detection (custom)
# -----------------------------

def sobel_edges(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    sy = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]], dtype=np.float32)

    def _single(ch):
        gx = _convolve2d(ch, sx, padding='reflect')
        gy = _convolve2d(ch, sy, padding='reflect')
        mag = np.sqrt(gx**2 + gy**2)
        return gx, gy, mag

    # Always compute on grayscale for gradient clarity
    if img.ndim == 3 and img.shape[2] >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    gx, gy, mag = _single(gray)
    gx_u8 = np.clip(np.abs(gx), 0, 255).astype(np.uint8)
    gy_u8 = np.clip(np.abs(gy), 0, 255).astype(np.uint8)
    mag_u8 = np.clip(mag, 0, 255).astype(np.uint8)
    return gx_u8, gy_u8, mag_u8


def prewitt_edges(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    px = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]], dtype=np.float32)
    py = np.array([[-1, -1, -1],
                   [0, 0, 0],
                   [1, 1, 1]], dtype=np.float32)

    def _single(ch):
        gx = _convolve2d(ch, px, padding='reflect')
        gy = _convolve2d(ch, py, padding='reflect')
        mag = np.sqrt(gx**2 + gy**2)
        return gx, gy, mag

    if img.ndim == 3 and img.shape[2] >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    gx, gy, mag = _single(gray)
    gx_u8 = np.clip(np.abs(gx), 0, 255).astype(np.uint8)
    gy_u8 = np.clip(np.abs(gy), 0, 255).astype(np.uint8)
    mag_u8 = np.clip(mag, 0, 255).astype(np.uint8)
    return gx_u8, gy_u8, mag_u8


def laplacian_edges(img: np.ndarray, kernel_type: str = '4') -> np.ndarray:
    if kernel_type == '8':
        k = np.array([[1, 1, 1],
                      [1, -8, 1],
                      [1, 1, 1]], dtype=np.float32)
    else:
        k = np.array([[0, -1, 0],
                      [-1, 4, -1],
                      [0, -1, 0]], dtype=np.float32)

    if img.ndim == 3 and img.shape[2] >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    res = _convolve2d(gray, k, padding='reflect')
    return np.clip(np.abs(res), 0, 255).astype(np.uint8)


def threshold(img: np.ndarray, t: int) -> np.ndarray:
    t = int(t)
    out = np.zeros_like(img, dtype=np.uint8)
    out[img >= t] = 255
    return out


def canny_edges(img: np.ndarray, t1: int = 100, t2: int = 200) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return cv2.Canny(gray, int(t1), int(t2))


# -----------------------------
# Dispatcher for API (custom)
# -----------------------------

def process_operation(img: np.ndarray, op: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process an operation and return a dict with results and optional intermediates for edges.
    """
    op = (op or '').lower()
    result: Dict[str, Any] = {}

    if op == 'grayscale':
        if img.ndim == 3 and img.shape[2] >= 3:
            res = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            res = img.copy()
        result['output'] = res

    # ----- Point processing (custom) -----
    elif op == 'point_negative':
        result['output'] = apply_negative_transform(img)

    elif op == 'point_log':
        result['output'] = apply_log_transform(img)

    elif op == 'point_power':
        gamma = float(params.get('gamma', 0.5))
        result['output'] = apply_power_transform(img, gamma)

    elif op == 'point_hist_eq':
        result['output'] = histogram_equalization(img)

    elif op == 'point_piecewise':
        r1 = int(params.get('r1', 100))
        s1 = int(params.get('s1', 0))
        r2 = int(params.get('r2', 200))
        s2 = int(params.get('s2', 255))
        result['output'] = apply_piecewise_linear(img, r1, s1, r2, s2)

    elif op == 'point_clahe':
        clip = float(params.get('clip_limit', 2.0))
        tiles = int(params.get('tiles', 8))
        result['output'] = custom_CLAHE(img, clip_limit=clip, tile_grid_size=(tiles, tiles))

    # ----- Noise/Blur/Sharpen/Edges (custom) -----
    elif op == 'noise_gaussian':
        mean = float(params.get('mean', 0.0))
        var = float(params.get('var', 10.0))
        result['output'] = add_gaussian_noise(img, mean, var)

    elif op == 'noise_sp':
        amount = float(params.get('amount', 0.02))
        s_vs_p = float(params.get('s_vs_p', 0.5))
        result['output'] = add_salt_pepper_noise(img, amount, s_vs_p)

    elif op == 'blur_mean':
        k = int(params.get('ksize', 3))
        result['output'] = mean_blur(img, k)

    elif op == 'blur_gaussian':
        k = int(params.get('ksize', 3))
        sigma = float(params.get('sigma', 1.0))
        result['output'] = gaussian_blur(img, k, sigma)

    elif op == 'blur_median':
        k = int(params.get('ksize', 3))
        result['output'] = median_blur(img, k)

    elif op == 'blur_bilateral':
        d = int(params.get('diameter', 9))
        sc = float(params.get('sigma_color', 75))
        ss = float(params.get('sigma_space', 75))
        result['output'] = bilateral_blur(img, d, sc, ss)

    elif op == 'sharpen_laplacian':
        alpha = float(params.get('alpha', 1.0))
        ktype = params.get('kernel_type', '4')
        result['output'] = laplacian_sharpen(img, alpha, ktype)

    elif op == 'sharpen_unsharp':
        k = int(params.get('ksize', 5))
        sigma = float(params.get('sigma', 1.0))
        amount = float(params.get('amount', 1.0))
        th = int(params.get('threshold', 0))
        result['output'] = unsharp_mask(img, k, sigma, amount, th)

    elif op == 'edge_sobel':
        gx, gy, mag = sobel_edges(img)
        t = int(params.get('threshold', -1))
        if t >= 0:
            bin_img = threshold(mag, t)
            result['threshold'] = bin_img
        result['grad_x'] = gx
        result['grad_y'] = gy
        result['magnitude'] = mag
        result['output'] = result.get('threshold', mag)

    elif op == 'edge_prewitt':
        gx, gy, mag = prewitt_edges(img)
        t = int(params.get('threshold', -1))
        if t >= 0:
            bin_img = threshold(mag, t)
            result['threshold'] = bin_img
        result['grad_x'] = gx
        result['grad_y'] = gy
        result['magnitude'] = mag
        result['output'] = result.get('threshold', mag)

    elif op == 'edge_laplacian':
        ktype = params.get('kernel_type', '4')
        lap = laplacian_edges(img, ktype)
        t = int(params.get('threshold', -1))
        if t >= 0:
            result['threshold'] = threshold(lap, t)
        result['laplacian'] = lap
        result['output'] = result.get('threshold', lap)

    elif op == 'edge_canny':
        t1 = int(params.get('t1', 100))
        t2 = int(params.get('t2', 200))
        result['output'] = canny_edges(img, t1, t2)

    else:
        raise ValueError(f"Unsupported operation: {op}")

    return result


# -----------------------------
# OpenCV-based implementations
# -----------------------------

def cv2_gaussian_noise(img: np.ndarray, mean: float = 0.0, var: float = 10.0) -> np.ndarray:
    def _fn(ch):
        noise = np.zeros_like(ch, dtype=np.int16)
        cv2.randn(noise, mean, np.sqrt(var))
        noisy = ch.astype(np.int16) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    return processing_image(img, _fn)


def cv2_salt_pepper_noise(img: np.ndarray, amount: float = 0.02, s_vs_p: float = 0.5) -> np.ndarray:
    # OpenCV doesn't have built-in S&P; use NumPy vectorization
    return add_salt_pepper_noise(img, amount, s_vs_p)


def cv2_mean_blur(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    ksize = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    return cv2.blur(img, (ksize, ksize))


def cv2_gaussian_blur(img: np.ndarray, ksize: int = 3, sigma: float = 1.0) -> np.ndarray:
    ksize = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=float(sigma), sigmaY=float(sigma))


def cv2_median_blur(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    ksize = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(img, ksize)


def cv2_bilateral_blur(img: np.ndarray, diameter: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    diameter = int(diameter) if int(diameter) > 0 else 9
    return cv2.bilateralFilter(img, diameter, float(sigma_color), float(sigma_space))


def cv2_laplacian_sharpen(img: np.ndarray, alpha: float = 1.0, ksize: int = 3) -> np.ndarray:
    ksize = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    if img.ndim == 3 and img.shape[2] >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    lap = cv2.Laplacian(gray, ddepth=cv2.CV_16S, ksize=ksize)
    lap = cv2.convertScaleAbs(lap)
    if gray.ndim == 2:
        sharp = cv2.subtract(gray, (alpha * lap).astype(np.uint8))
        return sharp
    else:
        return cv2.subtract(img, cv2.cvtColor((alpha * lap).astype(np.uint8), cv2.COLOR_GRAY2BGR))


def cv2_unsharp_mask(img: np.ndarray, ksize: int = 5, sigma: float = 1.0, amount: float = 1.0, threshold: int = 0) -> np.ndarray:
    blur = cv2_gaussian_blur(img, ksize, sigma)
    mask = cv2.subtract(img, blur)
    if threshold > 0:
        if mask.ndim == 3:
            low = (cv2.cvtColor(cv2.absdiff(img, blur), cv2.COLOR_BGR2GRAY) < threshold)
            mask[low] = 0
        else:
            low = (cv2.absdiff(img, blur) < threshold)
            mask[low] = 0
    return cv2.add(img, cv2.convertScaleAbs(mask, alpha=float(amount)))


def cv2_sobel_edges(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if img.ndim == 3 and img.shape[2] >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    gx = cv2.convertScaleAbs(gx)
    gy = cv2.convertScaleAbs(gy)
    mag = cv2.convertScaleAbs(mag)
    return gx, gy, mag


def cv2_prewitt_edges(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    kx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -1, -1],
                   [0, 0, 0],
                   [1, 1, 1]], dtype=np.float32)
    if img.ndim == 3 and img.shape[2] >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    gx = cv2.filter2D(gray, cv2.CV_32F, kx)
    gy = cv2.filter2D(gray, cv2.CV_32F, ky)
    mag = cv2.magnitude(gx, gy)
    gx = cv2.convertScaleAbs(gx)
    gy = cv2.convertScaleAbs(gy)
    mag = cv2.convertScaleAbs(mag)
    return gx, gy, mag


def cv2_laplacian_edges(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    ksize = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    if img.ndim == 3 and img.shape[2] >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=ksize)
    return cv2.convertScaleAbs(lap)


def process_operation_cv2(img: np.ndarray, op: str, params: Dict[str, Any]) -> Dict[str, Any]:
    op = (op or '').lower()
    result: Dict[str, Any] = {}

    if op == 'grayscale':
        out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if (img.ndim == 3 and img.shape[2] >= 3) else img.copy()
        result['output'] = out

    # ----- Point processing (cv2) -----
    elif op == 'point_negative':
        result['output'] = cv2.bitwise_not(img)

    elif op == 'point_log':
        # Use float, scale like cv2_apply_log_transform
        img_f = img.astype(np.float32)
        c = 255.0 / np.log(1 + np.max(img_f)) if np.max(img_f) > 0 else 1.0
        out = c * np.log(1 + img_f)
        result['output'] = np.clip(out, 0, 255).astype(np.uint8)

    elif op == 'point_power':
        gamma = float(params.get('gamma', 0.5))
        img_norm = img.astype(np.float32) / 255.0
        out = np.power(img_norm, gamma) * 255.0
        result['output'] = np.clip(out, 0, 255).astype(np.uint8)

    elif op == 'point_hist_eq':
        # Use color-aware equalization
        if img.ndim == 2:
            result['output'] = cv2.equalizeHist(img)
        else:
            b, g, r = cv2.split(img)
            result['output'] = cv2.merge((cv2.equalizeHist(b), cv2.equalizeHist(g), cv2.equalizeHist(r)))

    elif op == 'point_piecewise':
        r1 = int(params.get('r1', 100))
        s1 = int(params.get('s1', 0))
        r2 = int(params.get('r2', 200))
        s2 = int(params.get('s2', 255))
        # LUT approach
        lut = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            if i <= r1:
                lut[i] = int((s1 / max(r1, 1)) * i)
            elif i <= r2:
                lut[i] = int(((s2 - s1) / max((r2 - r1), 1)) * (i - r1) + s1)
            else:
                lut[i] = int(((255 - s2) / max((255 - r2), 1)) * (i - r2) + s2) if r2 != 255 else 255
        result['output'] = cv2.LUT(img, lut)

    elif op == 'point_clahe':
        clip = float(params.get('clip_limit', 2.0))
        tiles = max(2, int(params.get('tiles', 8)))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
        if img.ndim == 2:
            result['output'] = clahe.apply(img)
        else:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            result['output'] = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ----- Noise/Blur/Sharpen/Edges (cv2) -----
    elif op == 'noise_gaussian':
        mean = float(params.get('mean', 0.0))
        var = float(params.get('var', 10.0))
        result['output'] = cv2_gaussian_noise(img, mean, var)

    elif op == 'noise_sp':
        amount = float(params.get('amount', 0.02))
        s_vs_p = float(params.get('s_vs_p', 0.5))
        result['output'] = cv2_salt_pepper_noise(img, amount, s_vs_p)

    elif op == 'blur_mean':
        k = int(params.get('ksize', 3))
        result['output'] = cv2_mean_blur(img, k)

    elif op == 'blur_gaussian':
        k = int(params.get('ksize', 3))
        sigma = float(params.get('sigma', 1.0))
        result['output'] = cv2_gaussian_blur(img, k, sigma)

    elif op == 'blur_median':
        k = int(params.get('ksize', 3))
        result['output'] = cv2_median_blur(img, k)

    elif op == 'blur_bilateral':
        d = int(params.get('diameter', 9))
        sc = float(params.get('sigma_color', 75))
        ss = float(params.get('sigma_space', 75))
        result['output'] = cv2_bilateral_blur(img, d, sc, ss)

    elif op == 'sharpen_laplacian':
        alpha = float(params.get('alpha', 1.0))
        k = int(params.get('ksize', 3))
        result['output'] = cv2_laplacian_sharpen(img, alpha, k)

    elif op == 'sharpen_unsharp':
        k = int(params.get('ksize', 5))
        sigma = float(params.get('sigma', 1.0))
        amount = float(params.get('amount', 1.0))
        th = int(params.get('threshold', 0))
        result['output'] = cv2_unsharp_mask(img, k, sigma, amount, th)

    elif op == 'edge_sobel':
        gx, gy, mag = cv2_sobel_edges(img)
        t = int(params.get('threshold', -1))
        if t >= 0:
            result['threshold'] = threshold(mag, t)
        result['grad_x'] = gx
        result['grad_y'] = gy
        result['magnitude'] = mag
        result['output'] = result.get('threshold', mag)

    elif op == 'edge_prewitt':
        gx, gy, mag = cv2_prewitt_edges(img)
        t = int(params.get('threshold', -1))
        if t >= 0:
            result['threshold'] = threshold(mag, t)
        result['grad_x'] = gx
        result['grad_y'] = gy
        result['magnitude'] = mag
        result['output'] = result.get('threshold', mag)

    elif op == 'edge_laplacian':
        k = int(params.get('ksize', 3))
        lap = cv2_laplacian_edges(img, k)
        t = int(params.get('threshold', -1))
        if t >= 0:
            result['threshold'] = threshold(lap, t)
        result['laplacian'] = lap
        result['output'] = result.get('threshold', lap)

    elif op == 'edge_canny':
        t1 = int(params.get('t1', 100))
        t2 = int(params.get('t2', 200))
        result['output'] = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 and img.shape[2] >= 3 else img, t1, t2)

    else:
        raise ValueError(f"Unsupported operation: {op}")

    return result

