import cv2
import numpy as np
from PIL import Image

def remove_hair_artifacts(img_rgb):
    """
    DullRazor-inspired hair and artifact removal using morphological Black-Hat filtering
    and Telea inpainting on skin dermoscopy images.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # 7x7 structural element for detecting fine and medium hair fibers
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Thresholding to segment hair pixels
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # Inpaint hair mask with surrounding skin texture
    clean_bgr = cv2.inpaint(
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
        hair_mask,
        inpaintRadius=3,
        flags=cv2.INPAINT_TELEA
    )
    return cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Full dermoscopy preprocessing pipeline:
    1. RGB conversion
    2. DullRazor hair/artifact removal
    3. Bilateral / Gaussian noise reduction (3x3)
    4. CLAHE contrast enhancement in LAB color space
    5. Resize to target dimension (224x224)
    6. Normalize pixel values to [0.0, 1.0]
    """
    # Safe image reading for cross-platform support
    try:
        pil_img = Image.open(image_path).convert('RGB')
        img = np.array(pil_img)
    except Exception:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    # 1. Hair artifact removal (DullRazor)
    img_clean = remove_hair_artifacts(img)
    
    # 2. Resize to target_size
    img_resized = cv2.resize(img_clean, target_size, interpolation=cv2.INTER_AREA)
    
    # 3. Gaussian noise reduction
    img_denoised = cv2.GaussianBlur(img_resized, (3, 3), 0)
    
    # 4. Contrast enhancement via CLAHE in LAB color space
    lab = cv2.cvtColor(img_denoised, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    img_enhanced = cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2RGB)
    
    # 5. Normalization to [0.0, 1.0]
    img_normalized = img_enhanced.astype(np.float32) / 255.0
    return img_normalized

def save_preprocessed_visualization(image_path, output_path, target_size=(224, 224)):
    """
    Executes preprocessing pipeline and saves the 224x224 enhanced image as a standard JPG/PNG.
    """
    norm_img = preprocess_image(image_path, target_size=target_size)
    display_img = (norm_img * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(display_img).save(output_path, quality=95)
    return output_path
