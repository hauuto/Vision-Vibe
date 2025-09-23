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


class ImageFilters:
    """Optimized image filters implementation with better error handling"""

    @staticmethod
    def _validate_image(image: np.ndarray) -> bool:
        """Validate input image"""
        return image is not None and isinstance(image, np.ndarray) and image.size > 0

    @staticmethod
    def mean_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
        """Optimized mean blur using cv2 for better performance"""
        if not ImageFilters._validate_image(image):
            return image

        try:
            kernel_size = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
            result = cv2.blur(image, (kernel_size, kernel_size))
            return result if result is not None else image
        except Exception as e:
            print(f"Mean blur error: {e}")
            return image

    @staticmethod
    def gaussian_blur(image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
        """Optimized Gaussian blur with validation"""
        if not ImageFilters._validate_image(image):
            return image

        try:
            kernel_size = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
            sigma = max(0.1, sigma)  # Ensure positive sigma
            result = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
            return result if result is not None else image
        except Exception as e:
            print(f"Gaussian blur error: {e}")
            return image

    @staticmethod
    def median_filter(image: np.ndarray, kernel_size: int) -> np.ndarray:
        """Optimized median filter with validation"""
        if not ImageFilters._validate_image(image):
            return image

        try:
            kernel_size = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
            kernel_size = min(kernel_size, 31)  # OpenCV limit for median blur
            result = cv2.medianBlur(image, kernel_size)
            return result if result is not None else image
        except Exception as e:
            print(f"Median filter error: {e}")
            return image

    @staticmethod
    def bilateral_filter(image: np.ndarray, d: int, sigma_color: float, sigma_space: float) -> np.ndarray:
        """Optimized bilateral filter with validation"""
        if not ImageFilters._validate_image(image):
            return image

        try:
            d = max(5, min(d, 20))  # Reasonable range
            sigma_color = max(10, min(sigma_color, 200))
            sigma_space = max(10, min(sigma_space, 200))
            result = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
            return result if result is not None else image
        except Exception as e:
            print(f"Bilateral filter error: {e}")
            return image

    @staticmethod
    def sharpen_filter(image: np.ndarray) -> np.ndarray:
        """Sharpen filter using optimized kernel with validation"""
        if not ImageFilters._validate_image(image):
            return image

        try:
            kernel = np.array([[ 0, -1,  0],
                              [-1,  5, -1],
                              [ 0, -1,  0]], dtype=np.float32)
            result = cv2.filter2D(image, -1, kernel)
            return result if result is not None else image
        except Exception as e:
            print(f"Sharpen filter error: {e}")
            return image

    @staticmethod
    def unsharp_masking(image: np.ndarray, sigma: float = 1.0, strength: float = 1.5) -> np.ndarray:
        """Optimized unsharp masking with validation"""
        if not ImageFilters._validate_image(image):
            return image

        try:
            sigma = max(0.1, min(sigma, 10.0))
            strength = max(0.1, min(strength, 5.0))

            # Create blurred version
            blurred = cv2.GaussianBlur(image, (0, 0), sigma)
            if blurred is None:
                return image

            # Create and apply mask
            mask = cv2.subtract(image, blurred)
            sharpened = cv2.addWeighted(image, 1.0, mask, strength, 0)

            return np.clip(sharpened, 0, 255).astype(np.uint8) if sharpened is not None else image
        except Exception as e:
            print(f"Unsharp masking error: {e}")
            return image

    # Edge detection methods using OpenCV for better performance
    @staticmethod
    def sobel_x(image: np.ndarray) -> np.ndarray:
        """Optimized Sobel X edge detection with validation"""
        if not ImageFilters._validate_image(image):
            return image

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            result = np.uint8(np.absolute(sobel_x))
            return result if result is not None else image
        except Exception as e:
            print(f"Sobel X error: {e}")
            return image

    @staticmethod
    def sobel_y(image: np.ndarray) -> np.ndarray:
        """Optimized Sobel Y edge detection with validation"""
        if not ImageFilters._validate_image(image):
            return image

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            result = np.uint8(np.absolute(sobel_y))
            return result if result is not None else image
        except Exception as e:
            print(f"Sobel Y error: {e}")
            return image

    @staticmethod
    def sobel_magnitude(image: np.ndarray) -> np.ndarray:
        """Optimized Sobel magnitude with validation"""
        if not ImageFilters._validate_image(image):
            return image

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            result = np.uint8(np.clip(magnitude, 0, 255))
            return result if result is not None else image
        except Exception as e:
            print(f"Sobel magnitude error: {e}")
            return image

    @staticmethod
    def prewitt_edge(image: np.ndarray) -> np.ndarray:
        """Prewitt edge detection using custom kernels with validation"""
        if not ImageFilters._validate_image(image):
            return image

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

            kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
            kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

            prewitt_x = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
            prewitt_y = cv2.filter2D(gray, cv2.CV_32F, kernel_y)

            magnitude = np.sqrt(prewitt_x**2 + prewitt_y**2)
            result = np.uint8(np.clip(magnitude, 0, 255))
            return result if result is not None else image
        except Exception as e:
            print(f"Prewitt edge error: {e}")
            return image

    @staticmethod
    def laplacian_edge(image: np.ndarray) -> np.ndarray:
        """Optimized Laplacian edge detection with validation"""
        if not ImageFilters._validate_image(image):
            return image

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            result = np.uint8(np.absolute(laplacian))
            return result if result is not None else image
        except Exception as e:
            print(f"Laplacian edge error: {e}")
            return image

    @staticmethod
    def canny_edge(image: np.ndarray, low_threshold: int, high_threshold: int) -> np.ndarray:
        """Optimized Canny edge detection with validation"""
        if not ImageFilters._validate_image(image):
            return image

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            low_threshold = max(1, min(low_threshold, 255))
            high_threshold = max(low_threshold + 1, min(high_threshold, 255))
            result = cv2.Canny(gray, low_threshold, high_threshold)
            return result if result is not None else image
        except Exception as e:
            print(f"Canny edge error: {e}")
            return image

    @staticmethod
    def histogram_equalization(image: np.ndarray) -> np.ndarray:
        """Use optimized histogram equalization with validation"""
        if not ImageFilters._validate_image(image):
            return image

        try:
            result = histogram_equalization(image)
            return result if result is not None else image
        except Exception as e:
            print(f"Histogram equalization error: {e}")
            return image


class NoiseGenerator:
    """Optimized noise generation for testing with validation"""

    @staticmethod
    def add_gaussian_noise(image: np.ndarray, mean: float = 0, std: float = 25) -> np.ndarray:
        """Add Gaussian noise with proper type handling and validation"""
        if not ImageFilters._validate_image(image):
            return image

        try:
            mean = max(-50, min(mean, 50))
            std = max(1, min(std, 100))

            noise = np.random.normal(mean, std, image.shape).astype(np.float32)
            noisy = image.astype(np.float32) + noise
            result = np.clip(noisy, 0, 255).astype(np.uint8)
            return result
        except Exception as e:
            print(f"Gaussian noise error: {e}")
            return image

    @staticmethod
    def add_salt_pepper_noise(image: np.ndarray, prob: float = 0.05) -> np.ndarray:
        """Optimized salt and pepper noise with validation"""
        if not ImageFilters._validate_image(image):
            return image

        try:
            prob = max(0.001, min(prob, 0.2))  # Limit probability
            noisy_image = image.copy()
            total_pixels = image.shape[0] * image.shape[1]

            # Number of pixels to corrupt
            num_salt = int(prob * total_pixels * 0.5)
            num_pepper = int(prob * total_pixels * 0.5)

            if num_salt + num_pepper > 0:
                # Generate random coordinates
                coords = np.random.randint(0, image.shape[:2], size=(num_salt + num_pepper, 2))

                # Apply salt noise
                for i in range(num_salt):
                    row, col = coords[i]
                    if len(image.shape) == 3:
                        noisy_image[row, col] = [255, 255, 255]
                    else:
                        noisy_image[row, col] = 255

                # Apply pepper noise
                for i in range(num_salt, num_salt + num_pepper):
                    row, col = coords[i]
                    if len(image.shape) == 3:
                        noisy_image[row, col] = [0, 0, 0]
                    else:
                        noisy_image[row, col] = 0

            return noisy_image
        except Exception as e:
            print(f"Salt pepper noise error: {e}")
            return image


class ImageEnhancements:
    """Optimized image enhancement operations with validation"""

    @staticmethod
    def adjust_brightness(image: np.ndarray, value: int) -> np.ndarray:
        """Optimized brightness adjustment with validation"""
        if not ImageFilters._validate_image(image):
            return image

        try:
            value = max(-100, min(value, 100))  # Limit range
            result = cv2.convertScaleAbs(image, alpha=1, beta=value)
            return result if result is not None else image
        except Exception as e:
            print(f"Brightness adjustment error: {e}")
            return image

    @staticmethod
    def adjust_contrast(image: np.ndarray, alpha: float) -> np.ndarray:
        """Optimized contrast adjustment with validation"""
        if not ImageFilters._validate_image(image):
            return image

        try:
            alpha = max(0.1, min(alpha, 3.0))  # Limit range
            result = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
            return result if result is not None else image
        except Exception as e:
            print(f"Contrast adjustment error: {e}")
            return image

    @staticmethod
    def enhance_saturation(image: np.ndarray, factor: float = 1.3) -> np.ndarray:
        """Enhanced saturation adjustment using HSV with validation"""
        if not ImageFilters._validate_image(image):
            return image

        try:
            if len(image.shape) != 3:
                return image  # Can't enhance saturation of grayscale

            factor = max(0.1, min(factor, 3.0))  # Limit range
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = hsv[:, :, 1] * factor  # Enhance saturation
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)  # Clamp values
            hsv = hsv.astype(np.uint8)
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            return result if result is not None else image
        except Exception as e:
            print(f"Saturation enhancement error: {e}")
            return image
