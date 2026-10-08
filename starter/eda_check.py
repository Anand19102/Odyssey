#!/usr/bin/env python3
"""
starter/eda_check.py
Targeted EDA for Hackathon Scoring & Robustness Analysis
"""
import pandas as pd
import numpy as np

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

print("[*] Loading train and validation data for quick EDA...")
train = pd.read_csv("../data/train.csv", header=None, names=TRAIN_VAL_COLS, dtype=str, keep_default_na=False)
val = pd.read_csv("../data/validation.csv", header=None, names=TRAIN_VAL_COLS, dtype=str, keep_default_na=False)

print("\n" + "=" * 65)
print("1. CATEGORICAL CARDINALITY & UNSEEN VALUES (ROBUSTNESS CHECK)")
print("=" * 65)
for c in CATEGORICAL_COLS:
    tr_set = set(train[c].unique())
    val_set = set(val[c].unique())
    unseen_in_val = val_set - tr_set
    print(f"Feature: '{c}'")
    print(f"  Train unique: {len(tr_set)} | Val unique: {len(val_set)}")
    if unseen_in_val:
        print(f"  [!] Unseen categories in Validation: {unseen_in_val}")
    else:
        print(f"  [+] All categories in Validation appeared in Train.")

print("\n" + "=" * 65)
print("2. CASTING NUMERICS & CHECKING EXTREME SKEW / MAX VALUES")
print("=" * 65)
skewed_cols = []
for c in NUMERIC_COLS:
    tr_num = pd.to_numeric(train[c], errors="coerce").fillna(0.0)
    p50 = tr_num.median()
    p99 = tr_num.quantile(0.99)
    max_val = tr_num.max()
    # Check if max is 100x greater than 99th percentile (extreme heavy tail)
    if max_val > 0 and (max_val / (p99 + 1e-5)) > 50:
        skewed_cols.append((c, p50, p99, max_val))

print(f"{'Feature':<20} {'Median (50%)':<15} {'99th Percentile':<18} {'Max Value':<15}")
print("-" * 65)
for col, p50, p99, max_val in skewed_cols:
    print(f"{col:<20} {p50:<15.2f} {p99:<18.2f} {max_val:<15.2f}")

print("\n" + "=" * 65)
print("3. EXPLOITS vs NORMAL SIGNATURES (WHY EXPLOITS WERE MISSED)")
print("=" * 65)
train["Label_num"] = train["Label"].astype(int)
normal_mask = train["Label_num"] == 0
exploit_mask = train["attack_cat"].str.strip() == "Exploits"

# Compare key behavioral features between Normal and Exploits
compare_features = ["dur", "sbytes", "dbytes", "smeansz", "dmeansz", "tcprtt", "synack"]
print(f"{'Feature':<15} {'Normal (Median)':<20} {'Exploits (Median)':<20}")
print("-" * 65)
for cf in compare_features:
    norm_val = pd.to_numeric(train.loc[normal_mask, cf], errors="coerce").median()
    expl_val = pd.to_numeric(train.loc[exploit_mask, cf], errors="coerce").median()
    print(f"{cf:<15} {norm_val:<20.4f} {expl_val:<20.4f}")

print("=" * 65 + "\n")