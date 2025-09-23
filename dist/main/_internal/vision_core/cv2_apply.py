import cv2
import numpy as np

def cv2_apply_log_transform(image, c=None):
    """Log transformation using OpenCV operations"""
    # Convert to float for processing
    img_float = image.astype(np.float32)
    
    if c is None:
        c = 255.0 / np.log(1 + np.max(img_float))
    
    # Apply log transformation: s = c * log(1 + r)
    result = c * np.log(1 + img_float)
    
    # Clip and convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

def cv2_apply_power_transform(image, gamma=0.5, c=None):
    """Power transformation using OpenCV operations"""
    # Normalize to [0, 1]
    img_normalized = image.astype(np.float32) / 255.0
    
    if c is None:
        c = 1.0
    
    # Apply power transformation: s = c * r^gamma
    result = c * np.power(img_normalized, gamma)
    
    # Convert back to [0, 255]
    result = (result * 255).astype(np.uint8)
    return result

def cv2_apply_negative_transform(image):
    """Negative transformation using OpenCV operations"""
    # Simple bitwise NOT operation
    return cv2.bitwise_not(image)

def cv2_histogram_equalization(image):
    """Histogram equalization using OpenCV built-in functions - WITH NORMALIZATION"""
    print(f"🔧 OpenCV HE - Input range: [{np.min(image)}, {np.max(image)}]")
    
    if len(image.shape) == 2:  # Grayscale
        result = cv2.equalizeHist(image)
        
        # Normalize to full range if needed
        result_min = np.min(result)
        result_max = np.max(result)
        
        if result_max > result_min and (result_min > 0 or result_max < 255):
            result_normalized = ((result.astype(np.float64) - result_min) / (result_max - result_min) * 255)
            result = np.round(result_normalized).astype(np.uint8)
            print(f"🔧 OpenCV normalized range from [{result_min}, {result_max}] to [0, 255]")
        
        print(f"✅ OpenCV final range: [{np.min(result)}, {np.max(result)}]")
        return result
        
    elif len(image.shape) == 3 and image.shape[2] == 3:  # Color
        # Method 1: Equalize each channel separately (like custom method)
        b, g, r = cv2.split(image)
        
        # Apply histogram equalization to each channel
        b_eq = cv2.equalizeHist(b)
        g_eq = cv2.equalizeHist(g)
        r_eq = cv2.equalizeHist(r)
        
        # Normalize each channel if needed
        def normalize_channel(ch):
            ch_min = np.min(ch)
            ch_max = np.max(ch)
            if ch_max > ch_min and (ch_min > 0 or ch_max < 255):
                ch_normalized = ((ch.astype(np.float64) - ch_min) / (ch_max - ch_min) * 255)
                return np.round(ch_normalized).astype(np.uint8)
            return ch
        
        b_eq = normalize_channel(b_eq)
        g_eq = normalize_channel(g_eq)
        r_eq = normalize_channel(r_eq)
        
        result = cv2.merge((b_eq, g_eq, r_eq))
        print(f"✅ OpenCV final range: [{np.min(result)}, {np.max(result)}]")
        return result
    
    return image

def cv2_apply_piecewise_linear(image, r1=100, s1=0, r2=200, s2=255):
    """Piecewise linear transformation using OpenCV LUT"""
    # Create lookup table
    lut = np.zeros(256, dtype=np.uint8)
    
    for i in range(256):
        if i <= r1:
            lut[i] = int((s1 / r1) * i) if r1 != 0 else 0
        elif i <= r2:
            lut[i] = int(((s2 - s1) / (r2 - r1)) * (i - r1) + s1)
        else:
            lut[i] = int(((255 - s2) / (255 - r2)) * (i - r2) + s2) if r2 != 255 else 255
    
    # Ensure values are in valid range
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    
    # Apply LUT
    return cv2.LUT(image, lut)

def cv2_apply_CLAHE(image):
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

# Comparison functions
def compare_images(img1, img2):
    """Compare two images and return metrics"""
    try:
        # Ensure both images have the same shape
        if img1.shape != img2.shape:
            # Resize img2 to match img1 if different
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Calculate MSE
        mse = float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))
        
        # Calculate PSNR
        if mse < 1e-10:  # Very small MSE, essentially identical
            psnr = 100.0  # Set a high but finite value
        else:
            psnr = float(20 * np.log10(255.0 / np.sqrt(mse)))
        
        # Calculate SSIM (simplified version)
        def ssim_channel(img1_ch, img2_ch):
            # Convert to float64 for better precision
            img1_ch = img1_ch.astype(np.float64)
            img2_ch = img2_ch.astype(np.float64)
            
            mu1 = np.mean(img1_ch)
            mu2 = np.mean(img2_ch)
            sigma1_sq = np.var(img1_ch)
            sigma2_sq = np.var(img2_ch)
            sigma12 = np.mean((img1_ch - mu1) * (img2_ch - mu2))
            
            c1 = (0.01 * 255) ** 2
            c2 = (0.03 * 255) ** 2
            
            numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
            denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
            
            if denominator < 1e-10:
                return 1.0
            
            return numerator / denominator
        
        if len(img1.shape) == 2:
            ssim = ssim_channel(img1, img2)
        else:
            ssim_values = []
            for i in range(img1.shape[2]):
                ssim_val = ssim_channel(img1[:,:,i], img2[:,:,i])
                ssim_values.append(ssim_val)
            ssim = np.mean(ssim_values)
        
        # Ensure all values are finite and JSON-serializable
        return {
            'mse': round(max(0.0, min(mse, 999999.0)), 6),  # Cap MSE at reasonable value
            'psnr': round(max(0.0, min(psnr, 100.0)), 2),   # Cap PSNR at 100
            'ssim': round(max(0.0, min(float(ssim), 1.0)), 6)  # SSIM should be between 0 and 1
        }
        
    except Exception as e:
        print(f"Error in compare_images: {e}")
        # Return default values if comparison fails
        return {
            'mse': 0.0,
            'psnr': 0.0,
            'ssim': 0.0
        }