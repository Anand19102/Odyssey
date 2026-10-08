#!/usr/bin/env python3
"""
starter/model_v1.py
Experiment 1: Domain Feature Engineering + Regularized Tree Tuning
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, confusion_matrix
)

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


def load_dataset(filepath, is_labelled=True):
    print(f"[*] Loading dataset: {filepath} ...")
    cols = TRAIN_VAL_COLS if is_labelled else FEATURE_NAMES
    df = pd.read_csv(filepath, header=None, names=cols, dtype=str, keep_default_na=False)
    if is_labelled:
        df["attack_cat"] = df["attack_cat"].str.strip().replace({"": "Normal"})
        df["Label"] = pd.to_numeric(df["Label"], errors="coerce").fillna(0).astype(int)
    return df


def engineer_features(df, cat_levels=None, is_train=False):
    X = pd.DataFrame(index=df.index)

    # 1. Categorical handling (with consistent test-time mapping)
    if is_train or cat_levels is None:
        cat_levels = {}
        for c in CATEGORICAL_COLS:
            unique_vals = sorted(df[c].astype(str).unique())
            cat_levels[c] = {val: idx for idx, val in enumerate(unique_vals)}

    for c in CATEGORICAL_COLS:
        mapping = cat_levels[c]
        oov_idx = len(mapping)
        X[c] = df[c].astype(str).map(mapping).fillna(oov_idx).astype(int)

    # 2. Base numeric features cast to float32
    for c in NUMERIC_COLS:
        X[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)

    # 3. Domain Features: Traffic Ratios & Rates
    sbytes = X["sbytes"].values
    dbytes = X["dbytes"].values
    spkts = X["Spkts"].values
    dpkts = X["Dpkts"].values
    tcprtt = X["tcprtt"].values
    synack = X["synack"].values
    ackdat = X["ackdat"].values

    X["fe_bytes_ratio"] = np.float32(sbytes / (dbytes + 1.0))
    X["fe_pkts_ratio"] = np.float32(spkts / (dpkts + 1.0))
    X["fe_sbytes_per_pkt"] = np.float32(sbytes / (spkts + 1.0))
    X["fe_dbytes_per_pkt"] = np.float32(dbytes / (dpkts + 1.0))
    X["fe_synack_ratio"] = np.float32(synack / (tcprtt + 1e-6))
    X["fe_ackdat_ratio"] = np.float32(ackdat / (tcprtt + 1e-6))

    # 4. Log transforms for skewed volumes/delays
    log_candidates = ["dur", "sbytes", "dbytes", "Sload", "Dload", "Sintpkt", "Dintpkt"]
    for lc in log_candidates:
        X[f"fe_log_{lc}"] = np.log1p(np.maximum(X[lc].values, 0.0)).astype(np.float32)

    return X, cat_levels


def main():
    t0 = time.time()
    train_path = "../data/train.csv" if os.path.exists("../data/train.csv") else "data/train.csv"
    val_path = "../data/validation.csv" if os.path.exists("../data/validation.csv") else "data/validation.csv"

    train_df = load_dataset(train_path, is_labelled=True)
    val_df = load_dataset(val_path, is_labelled=True)

    print("[*] Generating domain engineered features ...")
    X_train, cat_levels = engineer_features(train_df, is_train=True)
    y_train = train_df["Label"].values

    X_val, _ = engineer_features(val_df, cat_levels=cat_levels, is_train=False)
    y_val = val_df["Label"].values

    cat_indices = [X_train.columns.get_loc(c) for c in CATEGORICAL_COLS]

    print(f"[*] Training tuned HistGradientBoostingClassifier ({X_train.shape[1]} features) ...")
    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.08,
        max_leaf_nodes=63,
        min_samples_leaf=40,
        l2_regularization=1.0,
        categorical_features=cat_indices,
        random_state=42
    )

    t_train = time.time()
    model.fit(X_train, y_train)
    print(f"    Finished training in {time.time() - t_train:.2f}s")

    y_val_prob = model.predict_proba(X_val)[:, 1]

    # Evaluate across thresholds
    thresholds = np.linspace(0.01, 0.90, 180)
    best_f1, best_f1_th = 0.0, 0.5
    for th in thresholds:
        f1 = f1_score(y_val, (y_val_prob >= th).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_f1_th = th

    best_cost, best_cost_th = float("inf"), 0.05
    for th in thresholds:
        y_pred = (y_val_prob >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()
        cost = 20.0 * fn + fp
        if cost < best_cost:
            best_cost = cost
            best_cost_th = th

    pr_auc = average_precision_score(y_val, y_val_prob)

    print("\n" + "=" * 70)
    print(" EXPERIMENT 1 VALIDATION RESULTS vs BASELINE")
    print("=" * 70)
    print(f"PR-AUC:          {pr_auc:.5f}  (Baseline: 0.99850)")
    print(f"Best F1 Score:   {best_f1:.5f} at Threshold {best_f1_th:.4f}  (Baseline: 0.9780)")
    print(f"Min Cost (w=20): {int(best_cost):,} at Threshold {best_cost_th:.4f}  (Baseline: 3,823)")
    print("=" * 70)

    # Save output submission file
    sub_df = pd.DataFrame({
        "row_id": np.arange(1, len(y_val_prob) + 1),
        "prediction": (y_val_prob >= best_f1_th).astype(int),
        "probability": np.round(y_val_prob, 6)
    })
    sub_df.to_csv("submission_validation_v1.csv", index=False)
    print(f"[*] Saved submission_validation_v1.csv ({len(sub_df):,} rows)")
    print(f"[+] Total execution time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()