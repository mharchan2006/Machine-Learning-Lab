import os
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from backend.config import PROCESSED_DIR, MODEL_PATH, RESULTS_DIR

def train_and_save_model():
    feat_path = os.path.join(PROCESSED_DIR, 'features.npy')
    label_path = os.path.join(PROCESSED_DIR, 'labels.npy')
    
    if not os.path.exists(feat_path) or not os.path.exists(label_path):
        raise FileNotFoundError("Processed features or labels not found. Please run generate_sample_dataset.py first.")
        
    X = np.load(feat_path)
    y = np.load(label_path)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    
    # MLP Classifier with Adam optimizer and L2 regularization
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate_init=0.01,
        warm_start=True,
        random_state=42
    )
    
    epochs = 15
    classes = np.array([0, 1])
    epoch_logs = []
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    
    for epoch in range(1, epochs + 1):
        model.partial_fit(X_train, y_train, classes=classes)
        
        train_prob = model.predict_proba(X_train)
        val_prob = model.predict_proba(X_val)
        
        t_acc = accuracy_score(y_train, train_prob.argmax(axis=1))
        v_acc = accuracy_score(y_val, val_prob.argmax(axis=1))
        t_loss = log_loss(y_train, train_prob)
        v_loss = log_loss(y_val, val_prob)
        
        train_accs.append(t_acc)
        val_accs.append(v_acc)
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        
        log_entry = {
            'epoch': epoch,
            'train_acc': round(float(t_acc) * 100, 1),
            'loss': round(float(t_loss), 4),
            'val_acc': round(float(v_acc) * 100, 1),
            'val_loss': round(float(v_loss), 4)
        }
        epoch_logs.append(log_entry)
        print(f"Epoch {epoch:2d}/{epochs}: Train Acc: {t_acc*100:5.1f}% | Loss: {t_loss:7.4f} | Val Acc: {v_acc*100:5.1f}%")
        
    # Save trained model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # Generate Training & Validation Curves plot
    os.makedirs(RESULTS_DIR, exist_ok=True)
    curve_path = os.path.join(RESULTS_DIR, 'training_curves.png')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    fig.patch.set_facecolor('#ffffff')
    
    ep_range = list(range(1, epochs + 1))
    
    # Accuracy Curve
    ax1.plot(ep_range, train_accs, 'o-', color='#2563eb', label='Training Accuracy', linewidth=2, markersize=5)
    ax1.plot(ep_range, val_accs, 's--', color='#dc2626', label='Validation Accuracy', linewidth=2, markersize=5)
    ax1.set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Accuracy', fontsize=10)
    ax1.set_ylim([0.45, 1.05])
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right', frameon=True)
    
    # Loss Curve
    ax2.plot(ep_range, train_losses, 'o-', color='#2563eb', label='Training Loss', linewidth=2, markersize=5)
    ax2.plot(ep_range, val_losses, 's--', color='#dc2626', label='Validation Loss', linewidth=2, markersize=5)
    ax2.set_title('Training & Validation Loss', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Loss', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    plt.savefig(curve_path, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {curve_path}")
    
    return {
        'status': 'success',
        'epochs': epochs,
        'final_val_acc': epoch_logs[-1]['val_acc'],
        'logs': epoch_logs,
        'curve_url': '/results/training_curves.png'
    }

if __name__ == '__main__':
    train_and_save_model()
