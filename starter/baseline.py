#!/usr/bin/env python3
"""
starter/baseline.py
===================
ML Hackathon  -  Network Intrusion Detection
Baseline Model Pipeline & Starter Solution

This script demonstrates an end-to-end intrusion detection solution:
1. Loads train.csv and validation.csv (with 38 features + 2 target labels).
2. Preprocesses categorical and numeric features cleanly.
3. Fits a fast, high-performance Gradient Boosting Classifier (HistGradientBoosting).
4. Evaluates predictions on validation.csv (F1, PR-AUC, Confusion Matrix).
5. Demonstrates threshold tuning to minimize asymmetric operational cost:
   Cost = weight * FN + FP.
6. Saves the trained model pipeline for easy submission and testing.
7. Generates a submission-ready predictions CSV formatted for evaluation.

Usage:
------
Run baseline training and validation evaluation:
    python baseline.py

Run on a quick subset (for fast testing/debugging on lower-spec hardware):
    python baseline.py --quick

Generate predictions on unlabelled challenge or test datasets:
    python baseline.py --predict path/to/test.csv --output submission_test.csv
"""

import os
import sys
import argparse
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, confusion_matrix
)

# ---------------------------------------------------------------------------
# Feature Column Definitions (matches docs/feature_dictionary.pdf)
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "proto", "state", "dur", "sbytes", "dbytes", "sloss", "dloss", "service",
    "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb",
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "Sjit", "Djit",
    "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports",
    "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst",
    "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm"
]
TRAIN_VAL_COLS = FEATURE_NAMES + ["attack_cat", "Label"]
CATEGORICAL_COLS = ["proto", "state", "service"]
NUMERIC_COLS = [c for c in FEATURE_NAMES if c not in CATEGORICAL_COLS]


def resolve_path(candidates):
    """Find the first existing path from a list of candidate locations."""
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return candidates[0] if candidates else ""


def load_dataset(filepath, is_labelled=True, nrows=None):
    """
    Load dataset from CSV. Handles headerless CSV files and assigns column names.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    print(f"[*] Loading dataset: {filepath} ...")
    cols = TRAIN_VAL_COLS if is_labelled else FEATURE_NAMES
    
    df = pd.read_csv(
        filepath,
        header=None,
        names=cols,
        dtype=str,
        keep_default_na=False,
        nrows=nrows
    )
    print(f"    Loaded {len(df):,} records with {df.shape[1]} columns.")
    
    if is_labelled:
        # Sanitize attack_cat (remove accidental leading/trailing whitespace like ' Fuzzers ')
        df["attack_cat"] = df["attack_cat"].str.strip().replace({"": "Normal"})
        df["Label"] = pd.to_numeric(df["Label"], errors="coerce").fillna(0).astype(int)
    
    return df


def preprocess_features(df, cat_levels=None, is_train=False):
    """
    Encode categorical columns and cast numeric columns to float32.
    Returns:
        X: Processed DataFrame
        cat_levels: Dictionary of categorical mappings
    """
    X = pd.DataFrame(index=df.index)
    if is_train or cat_levels is None:
        cat_levels = {}
        for c in CATEGORICAL_COLS:
            unique_vals = sorted(df[c].astype(str).unique())
            cat_levels[c] = {val: idx for idx, val in enumerate(unique_vals)}
    
    for c in CATEGORICAL_COLS:
        mapping = cat_levels[c]
        oov_idx = len(mapping)
        X[c] = df[c].astype(str).map(mapping).fillna(oov_idx).astype(int)
    
    for c in NUMERIC_COLS:
        X[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)
    
    return X, cat_levels


def evaluate_predictions(y_true, y_prob, threshold=0.5, cost_weight=20.0):
    """
    Compute binary classification metrics and asymmetric operational cost.
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = 0.0
    
    cost = cost_weight * fn + fp
    
    return {
        "threshold": threshold,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "pr_auc": pr_auc,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "cost": int(cost)
    }


def optimize_threshold_for_cost(y_true, y_prob, cost_weight=20.0, num_steps=200):
    """
    Scan probability thresholds to find the threshold minimizing:
        Cost = cost_weight * FN + FP
    """
    thresholds = np.linspace(0.005, 0.95, num_steps)
    best_thresh = 0.5
    lowest_cost = float("inf")
    best_metrics = None
    
    for th in thresholds:
        m = evaluate_predictions(y_true, y_prob, threshold=th, cost_weight=cost_weight)
        if m["cost"] < lowest_cost:
            lowest_cost = m["cost"]
            best_thresh = th
            best_metrics = m
            
    return best_thresh, best_metrics


def main():
    parser = argparse.ArgumentParser(description="Baseline Model for Network Intrusion Detection Hackathon")
    parser.add_argument("--train", type=str, default=None, help="Path to train.csv")
    parser.add_argument("--val", type=str, default=None, help="Path to validation.csv")
    parser.add_argument("--predict", type=str, default=None, help="Path to unlabelled test/challenge CSV to predict on")
    parser.add_argument("--output", type=str, default=None, help="Path to output submission CSV")
    parser.add_argument("--save-model", type=str, default="baseline_model.joblib", help="Path to save trained model artifact")
    parser.add_argument("--quick", action="store_true", help="Run on a small subset for fast verification")
    parser.add_argument("--cost-weight", type=float, default=20.0, help="False-negative penalty weight")
    args = parser.parse_args()

    # Determine default file locations
    train_candidates = [
        args.train or "",
        "data/train.csv",
        "../data/train.csv",
        os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")
    ]
    val_candidates = [
        args.val or "",
        "data/validation.csv",
        "../data/validation.csv",
        os.path.join(os.path.dirname(__file__), "..", "data", "validation.csv")
    ]
    train_path = resolve_path([p for p in train_candidates if p])
    val_path = resolve_path([p for p in val_candidates if p])

    print("=" * 70)
    print(" ML Hackathon  -  Network Intrusion Detection Starter Baseline")
    print("=" * 70)
    print(f"[*] Training dataset:   {train_path}")
    print(f"[*] Validation dataset: {val_path}")
    print(f"[*] Cost function:      Cost = {args.cost_weight:.0f} * FN + FP")
    if args.quick:
        print("[!] Quick mode enabled: Sampling 100,000 training rows.")
    print("-" * 70)

    # 1. Load Data
    t0 = time.time()
    train_df = load_dataset(train_path, is_labelled=True, nrows=100000 if args.quick else None)
    val_df = load_dataset(val_path, is_labelled=True, nrows=50000 if args.quick else None)

    # 2. Preprocess Features
    print("[*] Preprocessing features ...")
    X_train, cat_levels = preprocess_features(train_df, is_train=True)
    y_train = train_df["Label"].values
    
    X_val, _ = preprocess_features(val_df, cat_levels=cat_levels, is_train=False)
    y_val = val_df["Label"].values

    cat_feature_indices = [X_train.columns.get_loc(c) for c in CATEGORICAL_COLS]

    # 3. Model Training
    print("[*] Training HistGradientBoostingClassifier ...")
    model = HistGradientBoostingClassifier(
        max_iter=100 if not args.quick else 40,
        learning_rate=0.1,
        categorical_features=cat_feature_indices,
        random_state=42
    )
    t_train = time.time()
    model.fit(X_train, y_train)
    print(f"    Model training complete in {time.time() - t_train:.2f}s.")

    # 4. Evaluation on Validation Set
    print("\n" + "=" * 70)
    print(" VALIDATION EVALUATION RESULTS")
    print("=" * 70)
    y_val_prob = model.predict_proba(X_val)[:, 1]

    # Standard 0.50 threshold
    res_default = evaluate_predictions(y_val, y_val_prob, threshold=0.5, cost_weight=args.cost_weight)
    print(f"--- Standard Threshold (T = 0.50) ---")
    print(f"  F1-Score:     {res_default['f1']:.4f}")
    print(f"  PR-AUC:       {res_default['pr_auc']:.4f}")
    print(f"  Precision:    {res_default['precision']:.4f}")
    print(f"  Recall:       {res_default['recall']:.4f}")
    print(f"  Confusion:    TP={res_default['tp']:,} | FP={res_default['fp']:,} | TN={res_default['tn']:,} | FN={res_default['fn']:,}")
    print(f"  Total Cost:   {res_default['cost']:,}  (Cost = {args.cost_weight:.0f}*FN + FP)")

    # Cost-optimized threshold
    best_thresh, res_opt = optimize_threshold_for_cost(y_val, y_val_prob, cost_weight=args.cost_weight)
    cost_reduction = (1 - res_opt['cost'] / res_default['cost']) * 100
    print(f"\n--- Cost-Optimized Threshold (T* = {best_thresh:.4f}) ---")
    print(f"  F1-Score:     {res_opt['f1']:.4f}")
    print(f"  Precision:    {res_opt['precision']:.4f}")
    print(f"  Recall:       {res_opt['recall']:.4f} (Substantial increase in attack detection rate)")
    print(f"  Confusion:    TP={res_opt['tp']:,} | FP={res_opt['fp']:,} | TN={res_opt['tn']:,} | FN={res_opt['fn']:,}")
    print(f"  Total Cost:   {res_opt['cost']:,}  ({cost_reduction:.1f}% cost reduction)")

    # 5. Save Model Artifact for Reproducibility
    model_package = {
        "model": model,
        "cat_levels": cat_levels,
        "categorical_cols": CATEGORICAL_COLS,
        "numeric_cols": NUMERIC_COLS,
        "feature_names": FEATURE_NAMES,
        "best_threshold": float(best_thresh)
    }
    joblib.dump(model_package, args.save_model)
    print(f"\n[*] Model artifact saved for submission testing: {args.save_model}")

    # 6. Optional Prediction on Target Evaluation Set
    target_predict_path = args.predict
    output_sub_path = args.output or "submission_validation_baseline.csv"

    if target_predict_path:
        print(f"\n[*] Generating predictions for: {target_predict_path} ...")
        test_df = load_dataset(target_predict_path, is_labelled=False)
        X_test, _ = preprocess_features(test_df, cat_levels=cat_levels, is_train=False)
        test_probs = model.predict_proba(X_test)[:, 1]
        test_preds = (test_probs >= best_thresh).astype(int)
    else:
        test_probs = y_val_prob
        test_preds = (test_probs >= best_thresh).astype(int)
        print(f"\n[*] Generating sample submission for validation set: {output_sub_path}")

    sub_df = pd.DataFrame({
        "row_id": np.arange(1, len(test_preds) + 1),
        "prediction": test_preds,
        "probability": np.round(test_probs, 6)
    })
    sub_df.to_csv(output_sub_path, index=False)
    print(f"    Saved {len(sub_df):,} predictions to: {output_sub_path}")
    print(f"    First 5 submission lines:\n{sub_df.head(5).to_string(index=False)}")
    print("\n[+] Baseline pipeline finished successfully in {:.2f}s.".format(time.time() - t0))


if __name__ == "__main__":
    main()
