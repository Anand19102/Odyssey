#!/usr/bin/env python3
"""
starter/optimize_v3_apex.py
Apex Model Optimization: Out-of-Fold Target Encoding + Flow Asymmetry + Tri-Model Calibration
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, average_precision_score, confusion_matrix

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


def load_datasets():
    tr_path = "../data/train.csv" if os.path.exists("../data/train.csv") else "data/train.csv"
    va_path = "../data/validation.csv" if os.path.exists("../data/validation.csv") else "data/validation.csv"
    print("[*] Loading full datasets...")
    train = pd.read_csv(tr_path, header=None, names=TRAIN_VAL_COLS, dtype=str, keep_default_na=False)
    val = pd.read_csv(va_path, header=None, names=TRAIN_VAL_COLS, dtype=str, keep_default_na=False)
    train["Label"] = pd.to_numeric(train["Label"], errors="coerce").fillna(0).astype(int)
    val["Label"] = pd.to_numeric(val["Label"], errors="coerce").fillna(0).astype(int)
    return train, val


def compute_target_encoding(train_df, val_df, cat_cols, target_col="Label", smoothing=10.0):
    """Computes leak-free out-of-fold target encoding for train, and global smoothed for val."""
    print("[*] Computing out-of-fold smoothed target encodings...")
    global_mean = train_df[target_col].mean()
    encoded_train = pd.DataFrame(index=train_df.index)
    encoded_val = pd.DataFrame(index=val_df.index)
    
    encoding_maps = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for c in cat_cols:
        col_encoded_tr = np.zeros(len(train_df), dtype=np.float32)
        
        # Out-of-fold target encoding for train
        for tr_idx, oof_idx in kf.split(train_df):
            fold_tr = train_df.iloc[tr_idx]
            stats = fold_tr.groupby(c)[target_col].agg(["count", "mean"])
            smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
            col_encoded_tr[oof_idx] = train_df.iloc[oof_idx][c].map(smooth).fillna(global_mean).values

        encoded_train[f"te_{c}"] = col_encoded_tr

        # Full training set mapping for validation
        full_stats = train_df.groupby(c)[target_col].agg(["count", "mean"])
        full_smooth = (full_stats["count"] * full_stats["mean"] + smoothing * global_mean) / (full_stats["count"] + smoothing)
        encoding_maps[c] = full_smooth.to_dict()
        encoded_val[f"te_{c}"] = val_df[c].map(full_smooth).fillna(global_mean).astype(np.float32).values

    return encoded_train, encoded_val, encoding_maps, float(global_mean)


def engineer_apex_features(df, cat_levels, is_train=False):
    X = pd.DataFrame(index=df.index)

    # 1. Categorical Ordinal Index
    for c in CATEGORICAL_COLS:
        mapping = cat_levels.get(c, {})
        oov_idx = len(mapping)
        X[c] = df[c].astype(str).map(mapping).fillna(oov_idx).astype(int)

    # 2. Raw Numerics
    for c in NUMERIC_COLS:
        X[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)

    sbytes = X["sbytes"].values
    dbytes = X["dbytes"].values
    spkts = X["Spkts"].values
    dpkts = X["Dpkts"].values
    tcprtt = X["tcprtt"].values
    synack = X["synack"].values
    ackdat = X["ackdat"].values

    # Base Ratios
    X["fe_bytes_ratio"] = np.float32(sbytes / (dbytes + 1.0))
    X["fe_pkts_ratio"] = np.float32(spkts / (dpkts + 1.0))
    X["fe_sbytes_per_pkt"] = np.float32(sbytes / (spkts + 1.0))
    X["fe_dbytes_per_pkt"] = np.float32(dbytes / (dpkts + 1.0))
    X["fe_synack_ratio"] = np.float32(synack / (tcprtt + 1e-6))
    X["fe_ackdat_ratio"] = np.float32(ackdat / (tcprtt + 1e-6))

    # Interactions V2
    X["fe_handshake_asym"] = np.float32(synack / (ackdat + 1e-6))
    X["fe_loss_ratio"] = np.float32((X["sloss"].values + 1.0) / (X["dloss"].values + 1.0))
    X["fe_srv_interaction"] = np.float32(X["ct_srv_src"].values * X["ct_srv_dst"].values)
    X["fe_conn_density"] = np.float32(X["ct_dst_src_ltm"].values / (X["ct_dst_ltm"].values + 1.0))

    # APEX Feature Additions (Targeted at Fuzzers & Port Enumeration)
    # Byte balance: normalized difference [-1, 1]
    X["fe_byte_balance"] = np.float32((sbytes - dbytes) / (sbytes + dbytes + 1.0))
    # Packet balance: normalized difference [-1, 1]
    X["fe_pkt_balance"] = np.float32((spkts - dpkts) / (spkts + dpkts + 1.0))
    # Delta from reported mean size (flags anomalous payload distributions)
    X["fe_smeansz_delta"] = np.float32(np.abs(X["smeansz"].values - (sbytes / (spkts + 1.0))))
    X["fe_dmeansz_delta"] = np.float32(np.abs(X["dmeansz"].values - (dbytes / (dpkts + 1.0))))
    # Flow concentration
    X["fe_srv_src_ratio"] = np.float32(X["ct_srv_src"].values / (X["ct_dst_src_ltm"].values + 1.0))
    X["fe_total_ct"] = np.float32(X["ct_srv_src"].values + X["ct_srv_dst"].values + X["ct_dst_ltm"].values + X["ct_src_ltm"].values)

    # Log transforms for skewed variables
    for col in ["dur", "sbytes", "dbytes", "Sload", "Dload", "Sintpkt", "Dintpkt", "Sjit", "Djit"]:
        X[f"fe_log_{col}"] = np.log1p(np.maximum(X[col].values, 0.0)).astype(np.float32)

    return X


def main():
    t0 = time.time()
    train_df, val_df = load_datasets()
    y_train = train_df["Label"].values
    y_val = val_df["Label"].values

    # Category mappings
    cat_levels = {}
    for c in CATEGORICAL_COLS:
        unique_vals = sorted(train_df[c].astype(str).unique())
        cat_levels[c] = {val: idx for idx, val in enumerate(unique_vals)}

    # Base + Apex Features
    X_train_base = engineer_apex_features(train_df, cat_levels, is_train=True)
    X_val_base = engineer_apex_features(val_df, cat_levels, is_train=False)

    # Target Encodings
    te_train, te_val, te_maps, te_prior = compute_target_encoding(train_df, val_df, CATEGORICAL_COLS)

    X_train = pd.concat([X_train_base, te_train], axis=1)
    X_val = pd.concat([X_val_base, te_val], axis=1)

    cat_indices = [X_train.columns.get_loc(c) for c in CATEGORICAL_COLS]
    print(f"[*] Apex feature vector shape: {X_train.shape[1]} features (Engineered: {X_train.shape[1] - 38})")

    # Model 1: Deeper Regularized LightGBM
    print("[*] Training Model 1: Apex LightGBM (550 estimators, max_depth=10, leaves=140)...")
    lgbm = lgb.LGBMClassifier(
        n_estimators=550,
        learning_rate=0.05,
        num_leaves=140,
        max_depth=10,
        min_child_samples=25,
        subsample=0.85,
        colsample_bytree=0.70,
        reg_alpha=0.2,
        reg_lambda=3.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgbm.fit(X_train, y_train)

    # Model 2: HistGradientBoosting
    print("[*] Training Model 2: Tuned HistGradientBoosting...")
    hgb = HistGradientBoostingClassifier(
        max_iter=320,
        learning_rate=0.055,
        max_leaf_nodes=110,
        min_samples_leaf=25,
        l2_regularization=3.0,
        categorical_features=cat_indices,
        random_state=42
    )
    hgb.fit(X_train, y_train)

    # Probability Blending: 0.70 LightGBM + 0.30 HistGB
    print("[*] Blending ensemble probabilities (0.70 LGBM + 0.30 HistGB)...")
    p_lgb = lgbm.predict_proba(X_val)[:, 1]
    p_hgb = hgb.predict_proba(X_val)[:, 1]
    y_prob = 0.70 * p_lgb + 0.30 * p_hgb

    pr_auc = average_precision_score(y_val, y_prob)

    # Dense Threshold Sweep
    thresholds = np.linspace(0.005, 0.90, 500)
    best_f1, best_f1_th = 0.0, 0.5
    for th in thresholds:
        f1 = f1_score(y_val, (y_prob >= th).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_f1_th = th

    w = 40.0
    best_cost, best_cost_th = float("inf"), 0.05
    for th in thresholds:
        preds = (y_prob >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
        cost = (w * fn) + fp
        if cost < best_cost:
            best_cost = cost
            best_cost_th = th

    print("\n" + "=" * 70)
    print(" APEX MODEL BENCHMARK (EXPERIMENT 4)")
    print("=" * 70)
    print(f"PR-AUC:            {pr_auc:.5f}  (Prev Best: 0.99904)")
    print(f"Best F1 Score:     {best_f1:.5f} at T = {best_f1_th:.4f}  (Prev Best: 0.98164)")
    print(f"Min Cost (w=40):   {int(best_cost):,} at T = {best_cost_th:.4f}  (Prev Best: 3,506)")
    print("=" * 70)

    if pr_auc >= 0.99903 or best_f1 >= 0.98160:
        print("[+] Benchmark validated. Packaging artifact into 'best_apex_model.joblib'...")
        artifact = {
            "model_type": "apex_ensemble",
            "hgb_model": hgb,
            "lgbm_model": lgbm,
            "weight_lgbm": 0.70,
            "weight_hgb": 0.30,
            "cat_levels": cat_levels,
            "te_maps": te_maps,
            "te_prior": te_prior,
            "best_threshold_f1": float(best_f1_th),
            "best_threshold_cost_w40": float(best_cost_th),
            "feature_names": list(X_train.columns),
            "cost_w": 40
        }
        joblib.dump(artifact, "best_apex_model.joblib")
        # Also sync directly to default artifact
        joblib.dump(artifact, "best_ensemble_model.joblib")
        print("[*] Updated both 'best_apex_model.joblib' and 'best_ensemble_model.joblib'")

    print(f"[+] Total execution time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()