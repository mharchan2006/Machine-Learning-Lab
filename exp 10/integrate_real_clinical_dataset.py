import os
import shutil
import urllib.request
import numpy as np
from PIL import Image
from backend.config import DATASET_DIR, SAMPLES_DIR, PROCESSED_DIR, UPLOADS_DIR
from backend.ml.preprocessing import preprocess_image, save_preprocessed_visualization
from backend.ml.feature_extraction import extract_features
from populate_dataset_and_train import train_and_save_model
from evaluate_test_set import evaluate_test_dataset

# Confirmed Ground-Truth Clinical Cases from ISIC Archive
CLINICAL_NEVI = [
    'ISIC_0000000', 'ISIC_0000001', 'ISIC_0000003', 'ISIC_0000005',
    'ISIC_0000006', 'ISIC_0000007', 'ISIC_0000008', 'ISIC_0000009',
    'ISIC_0000010', 'ISIC_0000011', 'ISIC_0000012', 'ISIC_0000014',
    'ISIC_0000015', 'ISIC_0000016', 'ISIC_0000017', 'ISIC_0000018',
    'ISIC_0000019', 'ISIC_0000020'
]

CLINICAL_MELANOMAS = [
    'ISIC_0000002', 'ISIC_0000004', 'ISIC_0000013', 'ISIC_0000022',
    'ISIC_0000026', 'ISIC_0000029', 'ISIC_0000030', 'ISIC_0000031',
    'ISIC_0000035', 'ISIC_0000040'
]

def download_and_setup_clinical_dataset():
    desktop_clinical_dir = r'C:\Users\Hp\Desktop\Clinical_Dermoscopy_Samples'
    desktop_benign = os.path.join(desktop_clinical_dir, 'Benign_Nevi')
    desktop_melanoma = os.path.join(desktop_clinical_dir, 'Malignant_Melanoma')
    
    os.makedirs(desktop_benign, exist_ok=True)
    os.makedirs(desktop_melanoma, exist_ok=True)
    
    train_benign = os.path.join(DATASET_DIR, 'train', 'benign')
    train_mel = os.path.join(DATASET_DIR, 'train', 'melanoma')
    test_benign = os.path.join(DATASET_DIR, 'test', 'benign')
    test_mel = os.path.join(DATASET_DIR, 'test', 'melanoma')
    
    # 1. Download Clinical Benign Nevi
    print("Downloading authentic clinical Benign Nevi from ISIC Archive...")
    benign_paths = []
    for idx, isic_id in enumerate(CLINICAL_NEVI):
        url = f'https://isic-archive.s3.amazonaws.com/images/{isic_id}.jpg'
        dest_desktop = os.path.join(desktop_benign, f'Clinical_Benign_Nevus_{idx+1:02d}_{isic_id}.jpg')
        
        # Partition 12 for train, 6 for test
        if idx < 12:
            target_file = os.path.join(train_benign, f'{isic_id}.jpg')
        else:
            target_file = os.path.join(test_benign, f'{isic_id}.jpg')
            
        try:
            if not os.path.exists(target_file):
                urllib.request.urlretrieve(url, target_file)
            shutil.copyfile(target_file, dest_desktop)
            benign_paths.append(target_file)
            print(f"  [OK] Saved Benign: {isic_id}")
        except Exception as e:
            print(f"  [FAIL] Failed {isic_id}: {e}")
            
    # 2. Download Clinical Malignant Melanomas
    print("\nDownloading authentic clinical Malignant Melanomas from ISIC Archive...")
    mel_paths = []
    for idx, isic_id in enumerate(CLINICAL_MELANOMAS):
        url = f'https://isic-archive.s3.amazonaws.com/images/{isic_id}.jpg'
        dest_desktop = os.path.join(desktop_melanoma, f'Clinical_Melanoma_{idx+1:02d}_{isic_id}.jpg')
        
        # Partition 7 for train, 3 for test
        if idx < 7:
            target_file = os.path.join(train_mel, f'{isic_id}.jpg')
        else:
            target_file = os.path.join(test_mel, f'{isic_id}.jpg')
            
        try:
            if not os.path.exists(target_file):
                urllib.request.urlretrieve(url, target_file)
            shutil.copyfile(target_file, dest_desktop)
            mel_paths.append(target_file)
            print(f"  [OK] Saved Melanoma: {isic_id}")
        except Exception as e:
            print(f"  [FAIL] Failed {isic_id}: {e}")
            
    # 3. Setup Crisp Benchmark Samples
    clinical_sample_benign = os.path.join(train_benign, 'ISIC_0000000.jpg')
    clinical_sample_mel = os.path.join(train_mel, 'ISIC_0000002.jpg')
    
    shutil.copyfile(clinical_sample_benign, os.path.join(SAMPLES_DIR, 'benign_sample.jpg'))
    shutil.copyfile(clinical_sample_mel, os.path.join(SAMPLES_DIR, 'melanoma_sample.jpg'))
    
    shutil.copyfile(clinical_sample_benign, os.path.join(UPLOADS_DIR, 'benign_nevus_sample.jpg'))
    shutil.copyfile(clinical_sample_mel, os.path.join(UPLOADS_DIR, 'melanoma_dermoscopy_sample.jpg'))
    
    save_preprocessed_visualization(
        os.path.join(UPLOADS_DIR, 'benign_nevus_sample.jpg'),
        os.path.join(UPLOADS_DIR, 'prep_benign_nevus_sample.jpg')
    )
    save_preprocessed_visualization(
        os.path.join(UPLOADS_DIR, 'melanoma_dermoscopy_sample.jpg'),
        os.path.join(UPLOADS_DIR, 'prep_melanoma_dermoscopy_sample.jpg')
    )
    
    # 4. Extract Handcrafted Features for the Clinical Dataset
    print("\nExtracting 128-D features from real clinical dataset...")
    X, y = [], []
    
    for f in sorted(os.listdir(train_benign)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            p = os.path.join(train_benign, f)
            pre = preprocess_image(p)
            X.append(extract_features(pre))
            y.append(0)  # Benign
            
    for f in sorted(os.listdir(train_mel)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            p = os.path.join(train_mel, f)
            pre = preprocess_image(p)
            X.append(extract_features(pre))
            y.append(1)  # Melanoma
            
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    
    np.save(os.path.join(PROCESSED_DIR, 'features.npy'), X)
    np.save(os.path.join(PROCESSED_DIR, 'labels.npy'), y)
    print(f"Features updated: {X.shape}, labels: {y.shape}")
    
    # 5. Train MLP on Real Clinical Data
    print("\nTraining Multi-Layer Perceptron on Real Clinical Dermoscopy...")
    train_and_save_model()
    
    # 6. Evaluate on Unseen Clinical Test Set
    print("\nEvaluating on Unseen Clinical Test Set...")
    evaluate_test_dataset()
    
    print("\nAll done! Clinical samples saved to:", desktop_clinical_dir)

if __name__ == '__main__':
    download_and_setup_clinical_dataset()
