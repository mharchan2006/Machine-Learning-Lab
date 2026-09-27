import os
import random
import cv2
import numpy as np
from PIL import Image
from backend.config import DATASET_DIR, SAMPLES_DIR, PROCESSED_DIR
from backend.ml.preprocessing import preprocess_image
from backend.ml.feature_extraction import extract_features

def draw_hair_strands(img, num_hairs=12):
    """Adds realistic fine hair artifacts across dermoscopy image."""
    h, w, _ = img.shape
    for _ in range(num_hairs):
        # Spline / curved line for hair
        pt1 = (random.randint(0, w), random.randint(0, h))
        pt2 = (pt1[0] + random.randint(-80, 80), pt1[1] + random.randint(-80, 80))
        pt3 = (pt2[0] + random.randint(-80, 80), pt2[1] + random.randint(-80, 80))
        pts = np.array([pt1, pt2, pt3], np.int32).reshape((-1, 1, 2))
        thickness = random.randint(1, 2)
        hair_color = (random.randint(15, 45), random.randint(10, 35), random.randint(5, 25))
        cv2.polylines(img, [pts], isClosed=False, color=hair_color, thickness=thickness, lineType=cv2.LINE_AA)
    return img

def create_dermoscopy_lesion(is_melanoma=False, size=(300, 300)):
    w, h = size
    # Base skin color (warm peach/tan Fitzpatrick type II/III)
    base_color = np.array([random.randint(210, 235), random.randint(185, 205), random.randint(165, 185)], dtype=np.float32)
    img = np.ones((h, w, 3), dtype=np.float32) * base_color
    
    # Skin microtexture
    noise = np.random.normal(0, 5.0, (h, w, 3))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    center = (w // 2 + random.randint(-15, 15), h // 2 + random.randint(-15, 15))
    
    if not is_melanoma:
        # --- BENIGN NEVUS: Symmetric, smooth, uniform brown/tan ---
        rx = random.randint(55, 75)
        ry = int(rx * random.uniform(0.85, 1.15))
        angle = random.randint(0, 180)
        
        # Pigment lesion mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, center, (rx, ry), angle, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (25, 25), 0)
        
        # Uniform pigment color (warm dark brown/tan)
        lesion_color = np.array([random.randint(70, 95), random.randint(45, 65), random.randint(30, 48)], dtype=np.float32)
        lesion_layer = np.ones((h, w, 3), dtype=np.float32) * lesion_color
        
        # Blend
        alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
        img = (alpha * lesion_layer + (1 - alpha) * img.astype(np.float32)).clip(0, 255).astype(np.uint8)
        
    else:
        # --- MALIGNANT MELANOMA: Asymmetric, jagged/notched borders, variegated hues ---
        base_r = random.randint(65, 85)
        num_vertices = 24
        angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
        # Random radius variation creates irregular, notched borders
        radii = [base_r * random.uniform(0.55, 1.45) for _ in range(num_vertices)]
        
        pts = []
        for a, r in zip(angles, radii):
            px = int(center[0] + r * np.cos(a))
            py = int(center[1] + r * np.sin(a))
            pts.append([px, py])
            
        pts = np.array(pts, dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        
        # Secondary satellite lobules (common in invasive melanoma)
        for _ in range(random.randint(2, 4)):
            sat_center = (center[0] + random.randint(-base_r, base_r), center[1] + random.randint(-base_r, base_r))
            cv2.circle(mask, sat_center, random.randint(20, 38), 255, -1)
            
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        # Multi-hue variegated pigment (deep black, dark brown, reddish-pink, slate blue)
        c_black = np.array([25, 18, 15], dtype=np.float32)
        c_dark_brown = np.array([65, 35, 20], dtype=np.float32)
        c_erythema_red = np.array([160, 45, 45], dtype=np.float32)
        c_blue_veil = np.array([60, 65, 95], dtype=np.float32)
        
        # Create variegated lesion color field
        x_grid, y_grid = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
        var_field = (np.sin(x_grid * 3) + np.cos(y_grid * 3)) / 2.0
        
        lesion_layer = np.zeros((h, w, 3), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                v = var_field[i, j]
                if v < -0.3:
                    lesion_layer[i, j] = c_black
                elif v < 0.1:
                    lesion_layer[i, j] = c_dark_brown
                elif v < 0.4:
                    lesion_layer[i, j] = c_blue_veil
                else:
                    lesion_layer[i, j] = c_erythema_red
                    
        lesion_layer = cv2.GaussianBlur(lesion_layer, (19, 19), 0)
        alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
        img = (alpha * lesion_layer + (1 - alpha) * img.astype(np.float32)).clip(0, 255).astype(np.uint8)
        
    # Add realistic dermoscopy hair strands
    img = draw_hair_strands(img, num_hairs=random.randint(6, 14))
    
    # Subtle dermatoscope vignette (circular illumination)
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - w/2)**2 + (Y - h/2)**2)
    max_dist = np.sqrt((w/2)**2 + (h/2)**2)
    vignette = 1.0 - 0.25 * (dist_from_center / max_dist)**2
    img = (img * vignette[:, :, None]).clip(0, 255).astype(np.uint8)
    
    return img

def generate_all():
    print("Generating synthetic Dermoscopic Melanoma & Benign dataset...")
    
    train_benign_dir = os.path.join(DATASET_DIR, 'train', 'benign')
    train_melanoma_dir = os.path.join(DATASET_DIR, 'train', 'melanoma')
    test_benign_dir = os.path.join(DATASET_DIR, 'test', 'benign')
    test_melanoma_dir = os.path.join(DATASET_DIR, 'test', 'melanoma')
    
    # 1. Generate Training Set (35 benign, 35 melanoma)
    for i in range(1, 36):
        img_b = create_dermoscopy_lesion(is_melanoma=False)
        Image.fromarray(img_b).save(os.path.join(train_benign_dir, f'benign_train_{i:03d}.jpg'), quality=92)
        
        img_m = create_dermoscopy_lesion(is_melanoma=True)
        Image.fromarray(img_m).save(os.path.join(train_melanoma_dir, f'melanoma_train_{i:03d}.jpg'), quality=92)
        
    # 2. Generate Testing Set (15 benign, 15 melanoma)
    for i in range(1, 16):
        img_b = create_dermoscopy_lesion(is_melanoma=False)
        Image.fromarray(img_b).save(os.path.join(test_benign_dir, f'benign_test_{i:03d}.jpg'), quality=92)
        
        img_m = create_dermoscopy_lesion(is_melanoma=True)
        Image.fromarray(img_m).save(os.path.join(test_melanoma_dir, f'melanoma_test_{i:03d}.jpg'), quality=92)
        
    # 3. Generate Benchmark Sample Images
    sample_b = create_dermoscopy_lesion(is_melanoma=False, size=(380, 380))
    Image.fromarray(sample_b).save(os.path.join(SAMPLES_DIR, 'benign_sample.jpg'), quality=95)
    
    sample_m = create_dermoscopy_lesion(is_melanoma=True, size=(380, 380))
    Image.fromarray(sample_m).save(os.path.join(SAMPLES_DIR, 'melanoma_sample.jpg'), quality=95)
    
    print(f"Generated dataset in {DATASET_DIR}")
    print("Pre-extracting features for model training...")
    
    # Extract features for all training samples
    X = []
    y = []
    
    for f in os.listdir(train_benign_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            p = os.path.join(train_benign_dir, f)
            pre = preprocess_image(p)
            feats = extract_features(pre)
            X.append(feats)
            y.append(0)  # 0 = Benign
            
    for f in os.listdir(train_melanoma_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            p = os.path.join(train_melanoma_dir, f)
            pre = preprocess_image(p)
            feats = extract_features(pre)
            X.append(feats)
            y.append(1)  # 1 = Melanoma
            
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    
    np.save(os.path.join(PROCESSED_DIR, 'features.npy'), X)
    np.save(os.path.join(PROCESSED_DIR, 'labels.npy'), y)
    print(f"Saved processed features: shape {X.shape}, labels: shape {y.shape}")

if __name__ == '__main__':
    generate_all()
