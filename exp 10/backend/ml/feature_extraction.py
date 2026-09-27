import cv2
import numpy as np

def extract_features(img):
    """
    Extracts a 128-dimensional handcrafted dermoscopic feature vector:
    1. Colour Variegation: 32-bin normalized histograms for R, G, B channels (96 dimensions)
    2. Border Irregularity & Sobel Edge Energy: Gradients, edge magnitude statistics (12 dimensions)
    3. Structural Asymmetry: Horizontal, vertical, and quadrant disparity metrics (10 dimensions)
    4. Dermatoglyphic Texture & Contrast: High-frequency energy, color covariance (10 dimensions)
    Total: exactly 128 dimensions.
    """
    # Ensure float32 in [0, 1]
    if img.max() > 1.0:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
        
    features = []
    
    # --- 1. Color Variegation (96 features) ---
    for ch in range(3):
        channel_data = img[:, :, ch]
        hist, _ = np.histogram(channel_data, bins=32, range=(0.0, 1.0))
        hist = hist.astype(np.float32)
        norm_val = np.linalg.norm(hist)
        if norm_val > 0:
            hist /= norm_val
        features.extend(hist.tolist())
        
    # --- 2. Border Irregularity & Edge Gradients (12 features) ---
    gray = cv2.cvtColor((img * 255.0).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_energy = np.sqrt(gx**2 + gy**2)
    
    features.append(float(np.mean(edge_energy)))
    features.append(float(np.std(edge_energy)))
    features.append(float(np.var(edge_energy)))
    features.append(float(np.percentile(edge_energy, 50)))
    features.append(float(np.percentile(edge_energy, 75)))
    features.append(float(np.percentile(edge_energy, 90)))
    features.append(float(np.percentile(edge_energy, 98)))
    features.append(float(np.max(edge_energy)))
    
    # Laplacian of Gaussian (LoG) second derivative
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    features.append(float(np.mean(np.abs(laplacian))))
    features.append(float(np.std(laplacian)))
    features.append(float(np.max(np.abs(laplacian))))
    features.append(float(np.mean(laplacian**2)))
    
    # --- 3. Structural Asymmetry (10 features) ---
    h, w = gray.shape
    mid_h, mid_w = h // 2, w // 2
    
    left = gray[:, :mid_w]
    right = gray[:, mid_w:mid_w*2]
    h_asym = np.mean(np.abs(left - np.fliplr(right)))
    
    top = gray[:mid_h, :]
    bottom = gray[mid_h:mid_h*2, :]
    v_asym = np.mean(np.abs(top - np.flipud(bottom)))
    
    features.append(float(h_asym))
    features.append(float(v_asym))
    
    # 4 Quadrants disparity
    q1 = gray[:mid_h, :mid_w]
    q2 = gray[:mid_h, mid_w:]
    q3 = gray[mid_h:, :mid_w]
    q4 = gray[mid_h:, mid_w:]
    quad_means = [np.mean(q) for q in (q1, q2, q3, q4)]
    quad_stds = [np.std(q) for q in (q1, q2, q3, q4)]
    
    features.append(float(np.var(quad_means)))
    features.append(float(np.var(quad_stds)))
    features.extend([float(m) for m in quad_means])
    features.extend([float(s) for s in quad_stds[:2]])  # 2 more quadrant metrics
    
    # --- 4. Dermatoglyphic Texture & Color Covariance (10 features) ---
    # Inter-channel correlation/covariance
    r_ch = img[:, :, 0].flatten()
    g_ch = img[:, :, 1].flatten()
    b_ch = img[:, :, 2].flatten()
    
    cov_rg = float(np.corrcoef(r_ch, g_ch)[0, 1]) if np.std(r_ch) > 0 and np.std(g_ch) > 0 else 0.0
    cov_rb = float(np.corrcoef(r_ch, b_ch)[0, 1]) if np.std(r_ch) > 0 and np.std(b_ch) > 0 else 0.0
    cov_gb = float(np.corrcoef(g_ch, b_ch)[0, 1]) if np.std(g_ch) > 0 and np.std(b_ch) > 0 else 0.0
    
    features.append(cov_rg)
    features.append(cov_rb)
    features.append(cov_gb)
    
    # Texture roughness / local standard deviation
    blur = cv2.blur(gray, (5, 5))
    roughness = np.abs(gray - blur)
    features.append(float(np.mean(roughness)))
    features.append(float(np.std(roughness)))
    features.append(float(np.max(roughness)))
    
    # Energy and entropy approximations
    features.append(float(np.sum(gray**2) / (h * w)))
    features.append(float(np.std(gray)))
    features.append(float(np.percentile(gray, 25)))
    features.append(float(np.percentile(gray, 75)))
    
    # Ensure exact 128 dimensions
    feature_arr = np.array(features, dtype=np.float32)
    if len(feature_arr) < 128:
        feature_arr = np.pad(feature_arr, (0, 128 - len(feature_arr)), mode='constant')
    elif len(feature_arr) > 128:
        feature_arr = feature_arr[:128]
        
    # Replace any NaN/Inf with 0.0
    feature_arr = np.nan_to_num(feature_arr, nan=0.0, posinf=1.0, neginf=-1.0)
    return feature_arr
