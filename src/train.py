import os
import numpy as np
import copy
import time
from sklearn.metrics import roc_auc_score, f1_score, log_loss, confusion_matrix, precision_recall_curve

from dataset import CTRDataset
from model import MLP
from config import *

def get_lr(epoch):
    if epoch <= WARMUP_EPOCHS:
        return LEARNING_RATE * (epoch / WARMUP_EPOCHS)
    return LEARNING_RATE * (LR_DECAY_RATE ** (epoch - WARMUP_EPOCHS))

def find_best_threshold(y_true, y_pred):
    #Tìm threshold tối ưu cho F1
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx] if best_idx < len(thresholds) else 0.5, f1_scores[best_idx]

def evaluate_model(model, dataset, batch_size, desc="Eval"):
    #Đánh giá model trên dataset
    model.eval()
    all_preds, all_targets = [], []
    
    for X_b, y_b in dataset.batch_iterator(batch_size, shuffle=False):
        preds = model.predict(X_b)
        all_preds.extend(preds)
        all_targets.extend(y_b)
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Metrics
    auc = roc_auc_score(all_targets, all_preds)
    logloss = log_loss(all_targets, all_preds)
    best_th, best_f1 = find_best_threshold(all_targets, all_preds)
    
    return {
        'auc': auc,
        'logloss': logloss,
        'f1': best_f1,
        'threshold': best_th,
        'preds': all_preds,
        'targets': all_targets
    }

def train():
    print(f"TRAINING CONFIG")
    print(f"Model: {HIDDEN_DIMS} | Proj: {PROJ_DIM} | Dropout: {DROPOUT_RATE}")
    print(f"Batch: {BATCH_SIZE} | LR: {LEARNING_RATE} | WD: {WEIGHT_DECAY}")
    print(f"Early Stop: Patience={EARLY_STOPPING_PATIENCE}, Delta={EARLY_STOPPING_MIN_DELTA}")    
    # 1. Load Data
    train_ds = CTRDataset(os.path.join(PROCESSED_DATA_DIR, "train_X.npz"),
                          os.path.join(PROCESSED_DATA_DIR, "train_y.npy"), "train")
    val_ds = CTRDataset(os.path.join(PROCESSED_DATA_DIR, "val_X.npz"),
                        os.path.join(PROCESSED_DATA_DIR, "val_y.npy"), "val")
    
    # 2. Init Model
    model = MLP(input_dim=train_ds.X.shape[1], 
                hidden_dims=HIDDEN_DIMS, proj_dim=PROJ_DIM, 
                dropout=DROPOUT_RATE, weight_decay=WEIGHT_DECAY, 
                use_batch_norm=USE_BATCH_NORM)
    
    # 3. Training Loop
    best_logloss = np.inf
    best_model_weights = None
    patience = 0
    history = {'train_loss': [], 'val_auc': [], 'val_f1': []}
    
    print("\nStarting training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        start_t = time.time()
        curr_lr = get_lr(epoch)
        
        # --- TRAIN ---
        model.train()
        losses = []
        for X_b, y_b in train_ds.batch_iterator(BATCH_SIZE, shuffle=True):
            loss = model.backward(model.forward(X_b), y_b, curr_lr, GRADIENT_CLIP_VALUE)
            losses.append(loss)
        avg_loss = np.mean(losses)
        
        # --- VALIDATION ---
        val_metrics = evaluate_model(model, val_ds, BATCH_SIZE * 2, "Val")
        
        # Save history
        history['train_loss'].append(avg_loss)
        history['val_auc'].append(val_metrics['auc'])
        history['val_f1'].append(val_metrics['f1'])
        
        t_elapsed = time.time() - start_t
        
        # Print
        print(f"Epoch {epoch:02d} | LR: {curr_lr:.6f} | "
              f"Loss: {avg_loss:.4f} | "
              f"Val AUC: {val_metrics['auc']:.4f} | "
              f"Val F1: {val_metrics['f1']:.3f}@{val_metrics['threshold']:.2f} | "
              f"LogLoss: {val_metrics['logloss']:.4f} | "
              f"{t_elapsed:.1f}s")
        
        # --- EARLY STOPPING ---
        if val_metrics['logloss'] < best_logloss - EARLY_STOPPING_MIN_DELTA:
            best_logloss = val_metrics['logloss']
            best_model_weights = copy.deepcopy(model)
            patience = 0
            print(f"   New Best! (LogLoss: {best_logloss:.4f})")
        else:
            patience += 1
            print(f"  → Patience: {patience}/{EARLY_STOPPING_PATIENCE}")
            if patience >= EARLY_STOPPING_PATIENCE:
                print("\n  Early Stopping Triggered!")
                break
    
    # 4. FINAL TEST EVALUATION
    print("FINAL TEST EVALUATION")    
    if best_model_weights:
        model = best_model_weights
        print("Using best model from training")
    
    test_ds = CTRDataset(os.path.join(PROCESSED_DATA_DIR, "test_X.npz"),
                         os.path.join(PROCESSED_DATA_DIR, "test_y.npy"), "test")
    
    test_metrics = evaluate_model(model, test_ds, BATCH_SIZE * 2, "Test")
    
    # Detailed Results
    print(f"\n Test Metrics:")
    print(f"  AUC        : {test_metrics['auc']:.4f}")
    print(f"  LogLoss    : {test_metrics['logloss']:.4f}")
    print(f"  Best F1    : {test_metrics['f1']:.4f} (threshold: {test_metrics['threshold']:.3f})")
    
    # Confusion Matrix
    y_pred_bin = (test_metrics['preds'] > test_metrics['threshold']).astype(int)
    cm = confusion_matrix(test_metrics['targets'], y_pred_bin)
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\n Classification Report:")
    print(f"  Precision  : {precision:.4f} ({tp}/{tp + fp})")
    print(f"  Recall     : {recall:.4f} ({tp}/{tp + fn})")
    print(f"  Specificity: {tn/(tn+fp):.4f}")
    
    print(f"\n Confusion Matrix:")
    print(f"              Predicted")
    print(f"              Neg      Pos")
    print(f"  Actual Neg  {tn:<8} {fp:<8}")
    print(f"         Pos  {fn:<8} {tp:<8}")
    
    # Training Summary
    print(f"\n Training Summary:")
    print(f"  Best Val LogLoss: {best_logloss:.4f}")
    print(f"  Epochs Run  : {epoch}/{NUM_EPOCHS}")
    print(f"  Final Loss  : {avg_loss:.4f}")
    
    # Overfitting Check
    overfit_gap = val_metrics['auc'] - test_metrics['auc']  # Approximate using last val
    if abs(overfit_gap) > 0.02:
        if overfit_gap > 0:
            print(f"\nPossible overfitting (Val-Test gap: {overfit_gap:.4f})")
        else:
            print(f"\n Good generalization (Test AUC > Val AUC by {-overfit_gap:.4f})")
    else:
        print(f"\n  Good generalization (Val-Test gap: {overfit_gap:.4f})")
    
