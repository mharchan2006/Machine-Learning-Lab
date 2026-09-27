import os
import json
import urllib.request
from PIL import Image

def download_isic_samples():
    desktop_dir = r'C:\Users\Hp\Desktop\Clinical_Dermoscopy_Samples'
    benign_dir = os.path.join(desktop_dir, 'Benign_Nevi')
    melanoma_dir = os.path.join(desktop_dir, 'Malignant_Melanoma')
    
    os.makedirs(benign_dir, exist_ok=True)
    os.makedirs(melanoma_dir, exist_ok=True)
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    # 1. Fetch Benign Nevi from ISIC Archive
    print("Fetching authentic Benign Nevi from ISIC Archive...")
    url_b = 'https://api.isic-archive.com/api/v2/images/?diagnosis=nevus&limit=6'
    req_b = urllib.request.Request(url_b, headers=headers)
    with urllib.request.urlopen(req_b, timeout=15) as resp:
        data_b = json.loads(resp.read().decode('utf-8'))
        
    for idx, item in enumerate(data_b['results'], 1):
        isic_id = item['isic_id']
        img_url = item['files']['full']['url']
        out_path = os.path.join(benign_dir, f'Clinical_Benign_Nevus_{idx:02d}_{isic_id}.jpg')
        print(f"Downloading {isic_id} -> {out_path}...")
        urllib.request.urlretrieve(img_url, out_path)
        
    # 2. Fetch Malignant Melanoma from ISIC Archive
    print("Fetching authentic Malignant Melanoma from ISIC Archive...")
    url_m = 'https://api.isic-archive.com/api/v2/images/?diagnosis=melanoma&limit=6'
    req_m = urllib.request.Request(url_m, headers=headers)
    with urllib.request.urlopen(req_m, timeout=15) as resp:
        data_m = json.loads(resp.read().decode('utf-8'))
        
    for idx, item in enumerate(data_m['results'], 1):
        isic_id = item['isic_id']
        img_url = item['files']['full']['url']
        out_path = os.path.join(melanoma_dir, f'Clinical_Melanoma_{idx:02d}_{isic_id}.jpg')
        print(f"Downloading {isic_id} -> {out_path}...")
        urllib.request.urlretrieve(img_url, out_path)
        
    print("\nDownload complete! Files saved in:", desktop_dir)

if __name__ == '__main__':
    download_isic_samples()
