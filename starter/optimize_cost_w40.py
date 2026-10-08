#!/usr/bin/env python3
"""
starter/optimize_cost_w40.py
Hour 3 Cost Inject: Find Optimal Operating Threshold for w=40
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

print("[*] Loading best ensemble model artifact...")
package = joblib.load("best_ensemble_model.joblib")

# Read validation predictions from our ensemble run
val_preds_df = pd.read_csv("submission_validation_ensemble.csv")
val_probs = val_preds_df["probability"].values

# Read ground truth labels
print("[*] Loading ground truth labels from validation.csv...")
val_gt = pd.read_csv("../data/validation.csv", header=None, usecols=[39])
y_val = val_gt[39].astype(int).values

W_FN = 40.0
W_FP = 1.0

# Dense scan between 0.001 and 0.300
thresholds = np.linspace(0.005, 0.300, 300)
best_cost = float("inf")
best_th = 0.5
best_cm = None

for th in thresholds:
    preds = (val_probs >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
    cost = (W_FN * fn) + (W_FP * fp)
    if cost < best_cost:
        best_cost = cost
        best_th = th
        best_cm = (tn, fp, fn, tp)

tn, fp, fn, tp = best_cm
f1_at_cost = f1_score(y_val, (val_probs >= best_th).astype(int))

print("\n" + "=" * 65)
print(f"   HOUR 3 COST INJECT RESULTS (w={int(W_FN)})")
print("=" * 65)
print(f"Optimal Threshold (T*):      {best_th:.4f}")
print(f"Minimum Operational Cost:    {int(best_cost):,}")
print(f"F1-Score at this Threshold:  {f1_at_cost:.4f}")
print("-" * 65)
print("CONFUSION MATRIX BREAKDOWN:")
print(f"  True Positives  (TP caught):     {tp:,} / 44,040 ({tp/44040*100:.2f}%)")
print(f"  False Negatives (FN missed):     {fn:,} (Cost: {int(fn * W_FN):,})")
print(f"  False Positives (FP alarms):     {fp:,} (Cost: {int(fp * W_FP):,})")
print(f"  True Negatives  (TN passed):     {tn:,}")
print("=" * 65)

# Update artifact with the new w=40 threshold
package["best_threshold_cost_w40"] = float(best_th)
package["cost_w"] = 40
joblib.dump(package, "best_ensemble_model.joblib")
print("[+] Updated 'best_ensemble_model.joblib' with new cost threshold.")