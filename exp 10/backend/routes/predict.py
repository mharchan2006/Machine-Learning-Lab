import os
import shutil
import time
import joblib
from PIL import Image
from flask import Blueprint, request, jsonify
from backend.config import UPLOADS_DIR, SAMPLES_DIR, MODEL_PATH, CLASSES
from backend.ml.preprocessing import preprocess_image, save_preprocessed_visualization
from backend.ml.feature_extraction import extract_features
from backend.database import log_prediction

predict_bp = Blueprint('predict', __name__)

@predict_bp.route('/api/predict', methods=['POST'])
def predict():
    try:
        sample_type = request.form.get('sample')
        orig_filename = ""
        saved_path = ""
        
        if sample_type:
            # Benchmark sample selected
            if sample_type.lower() == 'melanoma':
                src_path = os.path.join(SAMPLES_DIR, 'melanoma_sample.jpg')
                orig_filename = "melanoma_dermoscopy_sample.jpg"
            else:
                src_path = os.path.join(SAMPLES_DIR, 'benign_sample.jpg')
                orig_filename = "benign_nevus_sample.jpg"
                
            timestamp = int(time.time() * 1000)
            saved_name = f"{timestamp}_{orig_filename}"
            saved_path = os.path.join(UPLOADS_DIR, saved_name)
            shutil.copyfile(src_path, saved_path)
            
        elif 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No image file uploaded'}), 400
            orig_filename = file.filename
            timestamp = int(time.time() * 1000)
            saved_name = f"{timestamp}_{orig_filename}"
            saved_path = os.path.join(UPLOADS_DIR, saved_name)
            file.save(saved_path)
        else:
            return jsonify({'error': 'No file uploaded or sample specified'}), 400
            
        # Inspect original image properties
        with Image.open(saved_path) as pil_im:
            orig_w, orig_h = pil_im.size
        file_size_kb = round(os.path.getsize(saved_path) / 1024, 1)
        
        # 1. Preprocess and create display image
        prep_filename = f"prep_{saved_name}"
        prep_path = os.path.join(UPLOADS_DIR, prep_filename)
        save_preprocessed_visualization(saved_path, prep_path, target_size=(224, 224))
        
        # 2. Extract features
        norm_img = preprocess_image(saved_path, target_size=(224, 224))
        features = extract_features(norm_img)
        
        # 3. Model inference
        if not os.path.exists(MODEL_PATH):
            return jsonify({'error': 'Model not trained yet. Please train the model.'}), 500
            
        model = joblib.load(MODEL_PATH)
        feat_vector = features.reshape(1, -1)
        proba = model.predict_proba(feat_vector)[0]
        
        benign_prob = round(float(proba[0]) * 100, 1)
        melanoma_prob = round(float(proba[1]) * 100, 1)
        pred_idx = int(proba.argmax())
        predicted_class = CLASSES[pred_idx]
        confidence = max(benign_prob, melanoma_prob)
        
        # 4. Log to SQLite database
        log_prediction(orig_filename, predicted_class, confidence)
        
        # Clinical advisory message
        if predicted_class == 'Melanoma':
            clinical_advice = "Asymmetric pigment architecture, irregular borders, and color variegation detected. Immediate dermatological evaluation & biopsy is recommended."
            is_malignant = True
        else:
            clinical_advice = "Dermoscopic pigment network and lesion boundary are within standard benign physiological parameters."
            is_malignant = False
            
        return jsonify({
            'success': True,
            'filename': orig_filename,
            'file_size_kb': file_size_kb,
            'resolution': f"{orig_w} × {orig_h} px",
            'original_image_url': f"/uploads/{saved_name}",
            'preprocessed_image_url': f"/uploads/{prep_filename}",
            'feature_length': len(features),
            'feature_sample': [round(float(v), 4) for v in features[:20]],
            'feature_vector_all': [round(float(v), 4) for v in features],
            'prediction': predicted_class,
            'confidence': confidence,
            'melanoma_prob': melanoma_prob,
            'benign_prob': benign_prob,
            'is_malignant': is_malignant,
            'clinical_advice': clinical_advice
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
