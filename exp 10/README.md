# Melanoma Skin Lesion Classification using Neural Network

Automated Dermoscopic Skin Lesion Analysis &bull; Benign Nevus vs. Malignant Melanoma Classification using a Multi-Layer Perceptron (MLP) Neural Network.

---

## 📌 Problem Statement
Melanoma is the most aggressive and lethal form of cutaneous malignancy, accounting for the vast majority of skin cancer deaths worldwide. When diagnosed and excised in its earliest localized stage (in situ or thin melanoma), the 5-year survival rate exceeds **98%**. However, once the malignant lesion penetrates deeper into the dermis and metastasizes, 5-year survival plummets to under **25%**. 

Clinical visual inspection by dermatologists using dermoscopy improves diagnostic accuracy, but remains labor-intensive, operator-dependent, subjective, and scarce in primary healthcare and underserved rural regions. 

To bridge this critical diagnostic gap, this project develops an automated, rapid, and accessible web-based computer-aided diagnosis (CAD) system. The application performs automated dermoscopy image preprocessing (including **DullRazor** morphological hair and artifact removal, and **CLAHE** contrast enhancement), extracts an ABCD rule-grounded **128-dimensional handcrafted feature vector** (color variegation histograms, Sobel edge gradients, structural asymmetry moments, and texture descriptors), and classifies the lesion using a **Multi-Layer Perceptron (MLP)** neural network with confidence probabilities, backed by SQLite diagnostic persistence.

---

## 🎯 Objectives
1. **Dataset Pipeline:** Collect, standardize, and preprocess high-resolution dermoscopic images into structured `train` and `test` directories (`Benign` vs. `Melanoma`).
2. **Preprocessing Pipeline:** Remove occluding hair artifacts using morphological Black-Hat filtering and inpainting (**DullRazor** algorithm), resize to 224&times;224, apply Gaussian denoising, and boost lesion contrast using **CLAHE** in LAB color space.
3. **Handcrafted Feature Engineering:** Extract a clinically grounded **128-dimensional feature vector** encoding the classical ABCD rule (Asymmetry, Border irregularity, Color variegation, and Dermatoglyphic Texture).
4. **Neural Network Classifier:** Design, train, optimize, and serialize a **Multi-Layer Perceptron (MLP)** classifier using Adam optimization and L2 regularization over 15 epochs.
5. **Interactive Single-Frame Web Application:** Build a responsive, dark-mode single-frame web interface (HTML5/CSS3/JavaScript) and Flask REST API integrating all 7 pipeline steps seamlessly.
6. **Quantitative Evaluation:** Evaluate the trained classifier on an unseen test cohort, computing Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix heatmap.
7. **Clinical Database Logging:** Persist every patient inference (image filename, predicted class, confidence percentage, timestamp) in an SQLite database (`database/melanoma.db`).

---

## 💻 Hardware & Software Requirements
- **Operating System:** Windows 10/11, macOS, or Linux
- **Language / Runtime:** Python 3.9+ (Tested on Python 3.10, 3.11, 3.12, 3.14)
- **Backend Framework:** Flask 3.x
- **Machine Learning Library:** scikit-learn (MLPClassifier, metrics)
- **Image Processing:** OpenCV (`opencv-python`), Pillow (`PIL`), NumPy
- **Plotting & Visualization:** Matplotlib (Agg backend)
- **Database:** SQLite3
- **Frontend Technologies:** Semantic HTML5, Modern CSS3 (CSS Grid & Flexbox, glowing themes), Vanilla JavaScript (ES6+), HTML5 Canvas

---

## 📂 Project Directory Structure

```text
MelanomaClassifier/
│
├── backend/
│   ├── dataset/
│   │   ├── samples/
│   │   │   ├── benign_sample.jpg
│   │   │   └── melanoma_sample.jpg
│   │   ├── test/
│   │   │   ├── benign/
│   │   │   └── melanoma/
│   │   └── train/
│   │       ├── benign/
│   │       └── melanoma/
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── feature_extraction.py
│   │   └── preprocessing.py
│   ├── models/
│   │   └── melanoma_model.pkl
│   ├── processed/
│   │   ├── features.npy
│   │   └── labels.npy
│   ├── results/
│   │   ├── confusion_matrix.png
│   │   └── training_curves.png
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── evaluate.py
│   │   ├── history.py
│   │   ├── predict.py
│   │   └── train.py
│   ├── uploads/
│   ├── app.py
│   ├── config.py
│   ├── database.py
│   └── requirements.txt
│
├── database/
│   └── melanoma.db
│
├── frontend/
│   ├── app.js
│   ├── index.html
│   └── style.css
│
├── evaluate_test_set.py
├── generate_sample_dataset.py
├── populate_dataset_and_train.py
└── README.md
```

---

## ⚙️ How to Compile and Run the Code

### Step 1: Open Terminal / Command Prompt
Navigate to the project root directory:
```bash
cd C:\Users\Hp\.gemini\antigravity\scratch\MelanomaClassifier
```

### Step 2: Install Required Dependencies
Ensure all required Python packages are installed:
```bash
python -m pip install -r backend/requirements.txt
```

### Step 3: Generate Dataset & Pre-extracted Features (If not already present)
```bash
python generate_sample_dataset.py
```

### Step 4: Train the Multi-Layer Perceptron Neural Network
```bash
python populate_dataset_and_train.py
```
*This trains the MLP model for 15 epochs, outputs validation scores, and generates `backend/results/training_curves.png`.*

### Step 5: Evaluate on the Test Set
```bash
python evaluate_test_set.py
```
*This calculates Accuracy, Precision, Recall, and F1-Score, and generates the Confusion Matrix heatmap.*

### Step 6: Launch the Web Application Server
```bash
python backend/app.py
```

### Step 7: Open in Web Browser
Open your preferred web browser and navigate to:
```text
http://127.0.0.1:5000
```
or
```text
http://localhost:5000
```

---

## 🔬 Clinical Diagnostic Workflow (All-in-One Frame)
1. **Load Image (Step 1):** Drag & drop a patient dermoscopy image or click either instant benchmark sample (`Load Malignant Melanoma Sample` or `Load Benign Nevus Sample`).
2. **Preprocessing (Step 2):** Inspect the side-by-side original image vs. DullRazor hair-filtered & CLAHE contrast-enhanced 224&times;224 view, along with the automated pipeline checklist.
3. **Feature Extraction (Step 3):** Inspect the exact 128-dimensional numerical feature vector sample and the dynamic Canvas spatial feature distribution chart.
4. **Model Training (Step 4):** Review the 15-epoch training logs, Training & Validation Accuracy and Loss curves, or trigger on-demand retraining with one click.
5. **Classification Result (Step 5):** Observe the pulsing 3D status orb (Red for Melanoma Detected, Green for Benign Nevus), confidence percentage, probability breakdown bars, and clinical advisory recommendation.
6. **Performance Evaluation (Step 6):** View the benchmark evaluation metric cards (Accuracy, Precision, Recall, F1-Score) and the Confusion Matrix heatmap.
7. **Database History (Step 7):** Review the real-time SQLite diagnostic table logging image filename, classification result, confidence score, and timestamp.
