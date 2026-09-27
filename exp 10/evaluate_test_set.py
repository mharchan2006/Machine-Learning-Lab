import os
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from backend.config import DATASET_DIR, MODEL_PATH, RESULTS_DIR
from backend.ml.preprocessing import preprocess_image
from backend.ml.feature_extraction import extract_features

def evaluate_test_dataset():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first.")
        
    model = joblib.load(MODEL_PATH)
    
    test_benign_dir = os.path.join(DATASET_DIR, 'test', 'benign')
    test_melanoma_dir = os.path.join(DATASET_DIR, 'test', 'melanoma')
    
    X_test = []
    y_test = []
    
    for f in sorted(os.listdir(test_benign_dir)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            p = os.path.join(test_benign_dir, f)
            pre = preprocess_image(p)
            X_test.append(extract_features(pre))
            y_test.append(0)  # Benign
            
    for f in sorted(os.listdir(test_melanoma_dir)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            p = os.path.join(test_melanoma_dir, f)
            pre = preprocess_image(p)
            X_test.append(extract_features(pre))
            y_test.append(1)  # Melanoma
            
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int64)
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot and save Confusion Matrix Heatmap
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    fig.patch.set_facecolor('#ffffff')
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    classes = ['Benign', 'Melanoma']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')
    ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold', pad=12)
    
    # Text annotations in each cell
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    fontsize=14, fontweight='bold',
                    color="white" if cm[i, j] > thresh else "black")
                    
    plt.tight_layout()
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")
    
    result = {
        'status': 'success',
        'accuracy': f"{acc * 100:.2f}%",
        'precision': f"{prec * 100:.2f}%",
        'recall': f"{rec * 100:.2f}%",
        'f1_score': f"{f1 * 100:.2f}%",
        'accuracy_num': round(float(acc) * 100, 2),
        'precision_num': round(float(prec) * 100, 2),
        'recall_num': round(float(rec) * 100, 2),
        'f1_score_num': round(float(f1) * 100, 2),
        'confusion_matrix': cm.tolist(),
        'cm_image_url': '/results/confusion_matrix.png'
    }
    print(f"Evaluation Complete: Accuracy={result['accuracy']}, Precision={result['precision']}, Recall={result['recall']}, F1={result['f1_score']}")
    return result

if __name__ == '__main__':
    evaluate_test_dataset()
