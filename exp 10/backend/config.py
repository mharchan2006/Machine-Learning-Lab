import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BACKEND_DIR = os.path.join(BASE_DIR, 'backend')
DATASET_DIR = os.path.join(BACKEND_DIR, 'dataset')
PROCESSED_DIR = os.path.join(BACKEND_DIR, 'processed')
MODELS_DIR = os.path.join(BACKEND_DIR, 'models')
RESULTS_DIR = os.path.join(BACKEND_DIR, 'results')
UPLOADS_DIR = os.path.join(BACKEND_DIR, 'uploads')
SAMPLES_DIR = os.path.join(DATASET_DIR, 'samples')
DATABASE_DIR = os.path.join(BASE_DIR, 'database')
DB_PATH = os.path.join(DATABASE_DIR, 'melanoma.db')
MODEL_PATH = os.path.join(MODELS_DIR, 'melanoma_model.pkl')
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')

CLASSES = ['Benign', 'Melanoma']
CLASS_DISPLAY = {
    'Benign': 'Benign Nevus (Healthy / Non-Cancerous)',
    'Melanoma': 'Malignant Melanoma Detected'
}

# Ensure all directories exist
for path in [PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, UPLOADS_DIR, SAMPLES_DIR, DATABASE_DIR]:
    os.makedirs(path, exist_ok=True)
