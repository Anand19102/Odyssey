#!/usr/bin/env python3
"""
starter/predict.py
==================
Standard Model Inference Script - Compatible with 66-Feature Apex Pipeline
Supports CLI evaluation:
    python predict.py --input <test.csv> --output <predictions.csv> [--model <model.joblib>] [--threshold <float>]
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "proto", "state", "dur", "sbytes", "dbytes", "sloss", "dloss", "service",
    "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb",
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "Sjit", "Djit",
    "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports",
    "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst",
    "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm"
]
CATEGORICAL_COLS = ["proto", "state", "service"]
NUMERIC_COLS = [c for c in FEATURE_NAMES if c not in CATEGORICAL_COLS]


def engineer_apex_features(df, cat_levels, te_maps=None, te_prior=0.115):
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

    # Interaction Terms
    X["fe_handshake_asym"] = np.float32(synack / (ackdat + 1e-6))
    X["fe_loss_ratio"] = np.float32((X["sloss"].values + 1.0) / (X["dloss"].values + 1.0))
    X["fe_srv_interaction"] = np.float32(X["ct_srv_src"].values * X["ct_srv_dst"].values)
    X["fe_conn_density"] = np.float32(X["ct_dst_src_ltm"].values / (X["ct_dst_ltm"].values + 1.0))

    # Apex Flow Terms
    X["fe_byte_balance"] = np.float32((sbytes - dbytes) / (sbytes + dbytes + 1.0))
    X["fe_pkt_balance"] = np.float32((spkts - dpkts) / (spkts + dpkts + 1.0))
    X["fe_smeansz_delta"] = np.float32(np.abs(X["smeansz"].values - (sbytes / (spkts + 1.0))))
    X["fe_dmeansz_delta"] = np.float32(np.abs(X["dmeansz"].values - (dbytes / (dpkts + 1.0))))
    X["fe_srv_src_ratio"] = np.float32(X["ct_srv_src"].values / (X["ct_dst_src_ltm"].values + 1.0))
    X["fe_total_ct"] = np.float32(X["ct_srv_src"].values + X["ct_srv_dst"].values + X["ct_dst_ltm"].values + X["ct_src_ltm"].values)

    # Log transforms
    for col in ["dur", "sbytes", "dbytes", "Sload", "Dload", "Sintpkt", "Dintpkt", "Sjit", "Djit"]:
        X[f"fe_log_{col}"] = np.log1p(np.maximum(X[col].values, 0.0)).astype(np.float32)

    # Smoothed Target Encodings
    if te_maps:
        for c in CATEGORICAL_COLS:
            mapping = te_maps.get(c, {})
            X[f"te_{c}"] = df[c].map(mapping).fillna(te_prior).astype(np.float32)

    return X


def main():
    parser = argparse.ArgumentParser(description="Network Intrusion Model Inference")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input unlabelled CSV (38 features)")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to save output prediction CSV")
    parser.add_argument("--model", "-m", type=str, default="best_ensemble_model.joblib", help="Path to model artifact")
    parser.add_argument("--threshold", "-t", type=float, default=None, help="Optional manual threshold override")
    args = parser.parse_args()

    model_path = args.model
    if not os.path.exists(model_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for cand in [os.path.join(script_dir, model_path), os.path.join(script_dir, "..", model_path)]:
            if os.path.exists(cand):
                model_path = cand
                break

    if not os.path.exists(model_path):
        print(f"[!] Error: Model artifact not found: {args.model}")
        sys.exit(1)

    print(f"[*] Loading model artifact: {model_path} ...")
    package = joblib.load(model_path)
    cat_levels = package.get("cat_levels", {})
    te_maps = package.get("te_maps", None)
    te_prior = package.get("te_prior", 0.115)

    if args.threshold is not None:
        threshold = args.threshold
    elif "best_threshold_cost_w40" in package:
        threshold = package["best_threshold_cost_w40"]
    else:
        threshold = package.get("best_threshold_f1", package.get("best_threshold", 0.5))

    print(f"[*] Reading test features: {args.input} ...")
    test_df = pd.read_csv(
        args.input,
        header=None,
        usecols=list(range(38)),
        names=FEATURE_NAMES,
        dtype=str,
        keep_default_na=False
    )
    print(f"    Loaded {len(test_df):,} records.")

    print("[*] Running Apex feature engineering (66 features) ...")
    X_test = engineer_apex_features(test_df, cat_levels, te_maps=te_maps, te_prior=te_prior)

    print(f"[*] Generating predictions (Operating Threshold = {threshold:.4f}) ...")
    if package.get("model_type") in ("ensemble", "apex_ensemble"):
        p_lgbm = package["lgbm_model"].predict_proba(X_test)[:, 1]
        p_hgb = package["hgb_model"].predict_proba(X_test)[:, 1]
        w_lgbm = package.get("weight_lgbm", 0.70)
        w_hgb = package.get("weight_hgb", 0.30)
        probs = w_lgbm * p_lgbm + w_hgb * p_hgb
    else:
        model = package["model"]
        probs = model.predict_proba(X_test)[:, 1]

    preds = (probs >= threshold).astype(int)

    sub_df = pd.DataFrame({
        "row_id": np.arange(1, len(preds) + 1),
        "prediction": preds,
        "probability": np.round(probs, 6)
    })
    sub_df.to_csv(args.output, index=False)
    print(f"[+] Output written to: {args.output}")
    print(f"    Total rows: {len(sub_df):,}")


if __name__ == "__main__":
    main()