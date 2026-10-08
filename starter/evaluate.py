#!/usr/bin/env python3
"""
starter/evaluate.py
===================
ML Hackathon  -  Network Intrusion Detection
Participant Submission Validator & Local Evaluator

This script allows participants to validate their submission files locally and
evaluate model performance on the validation dataset.

Evaluation Pipeline:
--------------------
submission.csv
      ↓
validate format (column names, row counts, row_id order, valid binary values, no NaNs)
      ↓
compare with validation ground truth labels
      ↓
calculate detection metrics (F1-Score, PR-AUC, Precision, Recall, Accuracy)
      ↓
calculate operational risk cost (Cost = weight * FN + FP)
      ↓
calculate per-attack-category recall breakdown
      ↓
generate evaluation report

Usage:
------
Validate and evaluate predictions on validation data:
    python evaluate.py --submission submission_validation.csv --ground-truth ../data/validation.csv

Validate submission file formatting only (without ground truth scoring):
    python evaluate.py --submission submission_challenge.csv --validate-only --expected-rows 254005
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, roc_auc_score,
    confusion_matrix, accuracy_score
)


def load_submission(filepath, expected_rows=None):
    """
    Validate and parse submission CSV file.
    Expected format:
        row_id,prediction[,probability]
        1,0,0.0125
        2,1,0.8920
    """
    if not os.path.exists(filepath):
        print(f"[!] ERROR: Submission file does not exist: {filepath}")
        sys.exit(1)

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"[!] ERROR: Failed to parse submission CSV: {e}")
        sys.exit(1)

    # Normalize column names to lowercase
    col_map = {c: str(c).strip().lower() for c in df.columns}
    df.rename(columns=col_map, inplace=True)

    # 1. Validation: Prediction column existence
    pred_col = None
    for cand in ["prediction", "pred", "label", "target"]:
        if cand in df.columns:
            pred_col = cand
            break

    if pred_col is None:
        if df.shape[1] == 1:
            pred_col = df.columns[0]
            print(f"[*] Notice: Single-column submission detected, using '{pred_col}' as prediction.")
        elif df.shape[1] >= 2 and "row_id" in df.columns:
            pred_col = [c for c in df.columns if c != "row_id"][0]
        else:
            print(f"[!] ERROR: Submission must contain a 'prediction' column. Found columns: {list(df.columns)}")
            sys.exit(1)

    # 2. Validation: Row count check (if expected_rows is provided)
    if expected_rows is not None and len(df) != expected_rows:
        print(f"[!] ERROR: Row count mismatch! Expected {expected_rows:,} rows, found {len(df):,} rows.")
        sys.exit(1)

    # 3. Validation: NaN / missing value checks
    if df[pred_col].isnull().any():
        nan_count = int(df[pred_col].isnull().sum())
        print(f"[!] ERROR: Submission contains {nan_count} missing (NaN) values in '{pred_col}'.")
        sys.exit(1)

    # 4. Validation: Binary integer values
    try:
        preds = df[pred_col].astype(float).values
    except ValueError:
        print(f"[!] ERROR: '{pred_col}' column contains non-numeric values.")
        sys.exit(1)

    invalid_mask = ~np.isin(preds, [0.0, 1.0])
    if invalid_mask.any():
        invalid_sample = preds[invalid_mask][:5]
        print(f"[!] ERROR: 'prediction' must be strictly binary (0 or 1). Found invalid values: {invalid_sample}")
        sys.exit(1)

    preds = preds.astype(int)

    # 5. Optional probability column
    prob_col = None
    probs = None
    for cand in ["probability", "prob", "score", "confidence"]:
        if cand in df.columns:
            prob_col = cand
            break

    if prob_col is not None:
        try:
            p_vals = df[prob_col].astype(float).values
            if (p_vals < 0.0).any() or (p_vals > 1.0).any():
                print(f"[!] WARNING: '{prob_col}' contains values outside [0.0, 1.0]. Clamping to range.")
                p_vals = np.clip(p_vals, 0.0, 1.0)
            probs = p_vals
        except ValueError:
            print(f"[!] WARNING: Could not parse '{prob_col}' as float probabilities. Skipping PR-AUC calculation.")

    # 6. Validation: row_id sequence (if present)
    if "row_id" in df.columns:
        row_ids = df["row_id"].values
        expected_ids = np.arange(1, len(df) + 1)
        if not np.array_equal(row_ids, expected_ids):
            print("[!] WARNING: 'row_id' does not match 1-based sequential order (1, 2, ..., N).")

    return {
        "df": df,
        "predictions": preds,
        "probabilities": probs,
        "count": len(df)
    }


def load_ground_truth(filepath):
    """
    Load ground truth labels from validation dataset (validation.csv).
    Handles 40-column headerless format where column 38 is attack_cat and column 39 is Label.
    """
    if not os.path.exists(filepath):
        print(f"[!] ERROR: Ground truth file does not exist: {filepath}")
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    if "Label" in first_line:
        df = pd.read_csv(filepath, keep_default_na=False)
        labels = df["Label"].astype(int).values
        attack_cats = df["attack_cat"].astype(str).str.strip().values if "attack_cat" in df.columns else None
    else:
        # Headerless dataset: column 38 is attack_cat, column 39 is Label
        df = pd.read_csv(filepath, header=None, dtype=str, keep_default_na=False, usecols=[38, 39])
        attack_cats = df[38].str.strip().replace({"": "Normal"}).values
        labels = df[39].astype(int).values

    return {
        "labels": labels,
        "attack_cats": attack_cats,
        "count": len(labels)
    }


def calculate_metrics(y_true, y_pred, y_prob=None, cost_weight=20.0):
    """
    Calculate intrusion detection performance metrics.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cost = cost_weight * fn + fp

    pr_auc = None
    roc_auc = None
    if y_prob is not None:
        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pr_auc = None
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = None
    else:
        try:
            pr_auc = float(average_precision_score(y_true, y_pred))
        except Exception:
            pr_auc = None

    return {
        "total_samples": len(y_true),
        "attacks_true": int(np.sum(y_true == 1)),
        "normal_true": int(np.sum(y_true == 0)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "cost": int(cost),
        "cost_weight": float(cost_weight)
    }


def print_evaluation_report(metrics, attack_cats=None, y_true=None, y_pred=None):
    """
    Print an evaluation report to stdout.
    """
    print("\n" + "=" * 74)
    print("                MODEL EVALUATION REPORT")
    print("=" * 74)
    print(f"Total Evaluated Records: {metrics['total_samples']:,} "
          f"(Normal: {metrics['normal_true']:,} | Attacks: {metrics['attacks_true']:,})")
    print("-" * 74)
    print("DETECTION PERFORMANCE METRICS:")
    print(f"  F1-Score:                {metrics['f1_score']:.4f}  (Harmonic mean of Prec & Rec)")
    pr_str = f"{metrics['pr_auc']:.4f}" if metrics['pr_auc'] is not None else "N/A (Provide probability column)"
    print(f"  PR-AUC:                  {pr_str}  (Precision-Recall Area Under Curve)")
    print(f"  Precision:               {metrics['precision']:.4f}  (Alert accuracy: TP / (TP + FP))")
    print(f"  Recall (Catch Rate):     {metrics['recall']:.4f}  (Detection rate: TP / (TP + FN))")
    print(f"  Accuracy (Reference):    {metrics['accuracy']:.4f}")

    print("\nCONFUSION MATRIX:")
    print(f"  True Positives  (TP):    {metrics['tp']:>8,}  [Correctly Identified Attacks]")
    print(f"  False Positives (FP):    {metrics['fp']:>8,}  [False Alarms on Normal Traffic]")
    print(f"  True Negatives  (TN):    {metrics['tn']:>8,}  [Legitimate Traffic Correctly Passed]")
    print(f"  False Negatives (FN):    {metrics['fn']:>8,}  [MISSED INTRUSIONS]")

    print(f"\nOPERATIONAL COST SCORE (Cost = {metrics['cost_weight']:.0f} * FN + FP):")
    fn_cost = int(metrics['cost_weight'] * metrics['fn'])
    print(f"  False Negative Loss:     {fn_cost:>8,}  ({metrics['fn']:,} missed attacks * {metrics['cost_weight']:.0f})")
    print(f"  False Alarm Overhead:    {metrics['fp']:>8,}  ({metrics['fp']:,} false alarms * 1)")
    print(f"  -------------------------------------")
    print(f"  TOTAL COST SCORE:        {metrics['cost']:>8,}  (Lower is better)")

    # Per-Attack Category Breakdown
    if attack_cats is not None and y_true is not None and y_pred is not None:
        print("\n" + "-" * 74)
        print("ATTACK CATEGORY DETECTION BREAKDOWN:")
        print(f"  {'Category':<18} {'Count':>8}   {'Metric':<20} {'Score':>8}")
        print("  " + "-" * 58)
        unique_cats = sorted(np.unique(attack_cats))
        for cat in unique_cats:
            mask = (attack_cats == cat)
            cnt = int(np.sum(mask))
            if cnt == 0:
                continue
            cat_pred = y_pred[mask]
            if cat == "Normal":
                spec = float(np.mean(cat_pred == 0))
                print(f"  {cat:<18} {cnt:>8,}   Specificity (Passed)   {spec:>7.2%}")
            else:
                rec = float(np.mean(cat_pred == 1))
                print(f"  {cat:<18} {cnt:>8,}   Recall (Detected)      {rec:>7.2%}")

    print("=" * 74 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Submission Validator & Evaluator")
    parser.add_argument("--submission", "-s", type=str, required=True, help="Path to submission CSV (row_id,prediction[,probability])")
    parser.add_argument("--ground-truth", "-g", type=str, default=None, help="Path to validation.csv ground truth")
    parser.add_argument("--validate-only", action="store_true", help="Only validate submission file format without scoring")
    parser.add_argument("--expected-rows", type=int, default=None, help="Expected number of submission rows (e.g. 254005 for challenge, 381007 for test)")
    parser.add_argument("--cost-weight", type=float, default=20.0, help="False-negative penalty weight (default: 20.0)")
    args = parser.parse_args()

    print("[*] Validating submission file: {} ...".format(args.submission))
    sub = load_submission(args.submission, expected_rows=args.expected_rows)
    print(f"    Submission format valid: {sub['count']:,} rows, binary predictions confirmed.")

    if args.validate_only or args.ground_truth is None:
        print("[+] Format validation successful! File is ready for submission.")
        return

    print("[*] Loading ground truth from: {} ...".format(args.ground_truth))
    gt = load_ground_truth(args.ground_truth)
    print(f"    Loaded {gt['count']:,} ground truth records.")

    if sub["count"] != gt["count"]:
        print(f"[!] FATAL ERROR: Row count mismatch!")
        print(f"    Submission has:   {sub['count']:,} rows")
        print(f"    Ground truth has: {gt['count']:,} rows")
        sys.exit(1)

    metrics = calculate_metrics(
        y_true=gt["labels"],
        y_pred=sub["predictions"],
        y_prob=sub["probabilities"],
        cost_weight=args.cost_weight
    )

    print_evaluation_report(
        metrics=metrics,
        attack_cats=gt["attack_cats"],
        y_true=gt["labels"],
        y_pred=sub["predictions"]
    )


if __name__ == "__main__":
    main()
