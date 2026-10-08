#!/usr/bin/env python3
"""
starter/ensemble.py
Experiment 3: Weighted Soft-Voting Blend (LightGBM + HistGradientBoosting)
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
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

    # Model 1: HistGB
    print("[*] Training Model 1: HistGradientBoosting ...")
    hgb = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.08,
        max_leaf_nodes=63,
        min_samples_leaf=40,
        l2_regularization=1.0,
        categorical_features=cat_indices,
        random_state=42
    )
    hgb.fit(X_train, y_train)

    # Model 2: LightGBM
    print("[*] Training Model 2: LightGBM ...")
    lgbm = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.08,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgbm.fit(X_train, y_train)

    # Soft Blend (0.6 LGBM + 0.4 HistGB)
    print("[*] Blending probabilities (0.60 LGBM + 0.40 HistGB) ...")
    p_hgb = hgb.predict_proba(X_val)[:, 1]
    p_lgbm = lgbm.predict_proba(X_val)[:, 1]
    y_val_prob = 0.60 * p_lgbm + 0.40 * p_hgb

    # Threshold scans
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
    print(" EXPERIMENT 3 (SOFT-VOTING ENSEMBLE) VALIDATION RESULTS")
    print("=" * 70)
    print(f"PR-AUC:          {pr_auc:.5f}  (LightGBM: 0.99901 | Baseline: 0.99850)")
    print(f"Best F1 Score:   {best_f1:.5f} at Threshold {best_f1_th:.4f}  (LightGBM: 0.98114)")
    print(f"Min Cost (w=20): {int(best_cost):,} at Threshold {best_cost_th:.4f}  (HistGB: 3,410 | LGBM: 3,454)")
    print("=" * 70)

    # Save Ensemble artifact bundle
    artifact = {
        "model_type": "ensemble",
        "hgb_model": hgb,
        "lgbm_model": lgbm,
        "weight_lgbm": 0.60,
        "weight_hgb": 0.40,
        "cat_levels": cat_levels,
        "best_threshold_f1": float(best_f1_th),
        "best_threshold_cost": float(best_cost_th),
        "feature_names": list(X_train.columns)
    }
    joblib.dump(artifact, "best_ensemble_model.joblib")
    print("[*] Saved artifact: best_ensemble_model.joblib")

    # Save validation predictions for evaluator check
    sub_df = pd.DataFrame({
        "row_id": np.arange(1, len(y_val_prob) + 1),
        "prediction": (y_val_prob >= best_f1_th).astype(int),
        "probability": np.round(y_val_prob, 6)
    })
    sub_df.to_csv("submission_validation_ensemble.csv", index=False)
    print(f"[*] Saved submission_validation_ensemble.csv ({len(sub_df):,} rows)")
    print(f"[+] Total execution time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()