"""
XGBoost attack classifier for headerless CSVs (last column = label, 0/1).

Usage:
    Edit the variables in the `if __name__ == "__main__"` block at the bottom,
    then run:  python xgb_attack.py

Or import and use from Python:
    from xgb_attack import train, evaluate_file
    train("train.csv", out_dir="model_dir")
    evaluate_file("validation.csv", model_dir="model_dir")

Requires: pip install xgboost pandas scikit-learn numpy
"""
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

MODEL_FILE = "model.json"
META_FILE = "meta.json"


# --------------------------------------------------------------------------
# Data loading / preprocessing
# --------------------------------------------------------------------------
# attack_cat-style column that leaks the label (second-to-last column). Dropped on load.
DROP_SECOND_LAST_COL = True


def load_raw(path: str) -> pd.DataFrame:
    """Read a headerless CSV. Columns are named c0..c{n-2}, last is 'label'."""
    df = pd.read_csv(path, header=None, skipinitialspace=True)
    if DROP_SECOND_LAST_COL:
        df = df.drop(columns=df.columns[-2])
    df.columns = [f"c{i}" for i in range(df.shape[1] - 1)] + ["label"]
    return df


def fit_preprocessor(df: pd.DataFrame) -> dict:
    """Learn which columns are categorical (strings) and their category lists."""
    feats = df.drop(columns="label")
    cat_cols = [c for c in feats.columns if feats[c].dtype == object]
    categories = {c: sorted(feats[c].astype(str).str.strip().unique().tolist())
                  for c in cat_cols}
    return {"columns": feats.columns.tolist(), "categories": categories}


def transform(df: pd.DataFrame, meta: dict):
    """Apply the learned preprocessing. Unseen categories become NaN."""
    cols = meta["columns"]
    if df.shape[1] - 1 != len(cols):
        raise ValueError(
            f"Column mismatch: file has {df.shape[1] - 1} feature columns, "
            f"model expects {len(cols)}."
        )
    X = df.drop(columns="label").copy()
    X.columns = cols
    for c in cols:
        if c in meta["categories"]:
            X[c] = pd.Categorical(X[c].astype(str).str.strip(),
                                  categories=meta["categories"][c])
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    y = df["label"].astype(int).values
    return X, y


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def compute_metrics(y_true, proba, threshold=0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    m = {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }
    if len(np.unique(y_true)) == 2:
        m["roc_auc"] = roc_auc_score(y_true, proba)
        m["pr_auc"] = average_precision_score(y_true, proba)
    m["confusion_matrix"] = confusion_matrix(y_true, pred, labels=[0, 1]).tolist()
    return m


def print_metrics(y_true, proba, threshold=0.5, title="Evaluation"):
    m = compute_metrics(y_true, proba, threshold)
    pred = (proba >= threshold).astype(int)
    print(f"\n===== {title} (threshold={threshold}) =====")
    for k, v in m.items():
        if k != "confusion_matrix":
            print(f"{k:>10}: {v:.4f}")
    tn, fp, fn, tp = np.array(m["confusion_matrix"]).ravel()
    print("\nConfusion matrix (rows=actual, cols=predicted):")
    print(f"            pred 0   pred 1\n  actual 0  {tn:7d}  {fp:7d}\n  actual 1  {fn:7d}  {tp:7d}")
    print("\n" + classification_report(y_true, pred, target_names=["normal", "attack"],
                                       zero_division=0))
    return m


# --------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------
def train(train_path, val_path=None, out_dir="model_dir", threshold=0.5,
          n_estimators=1000, learning_rate=0.05, max_depth=8, seed=42):
    os.makedirs(out_dir, exist_ok=True)

    df = load_raw(train_path)
    meta = fit_preprocessor(df)
    X, y = transform(df, meta)

    # Early-stopping set: use val file if given, else hold out 10% of train
    if val_path:
        Xv, yv = transform(load_raw(val_path), meta)
        X_tr, y_tr = X, y
    else:
        X_tr, Xv, y_tr, yv = train_test_split(
            X, y, test_size=0.1, stratify=y, random_state=seed)

    # Handle class imbalance
    pos, neg = (y_tr == 1).sum(), (y_tr == 0).sum()
    spw = float(neg / max(pos, 1))

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(Xv, yv)], verbose=50)

    model.save_model(os.path.join(out_dir, MODEL_FILE))
    meta["best_iteration"] = int(model.best_iteration)
    with open(os.path.join(out_dir, META_FILE), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved model + metadata to '{out_dir}/'")

    print_metrics(y_tr, model.predict_proba(X_tr)[:, 1], threshold, "TRAIN")
    print_metrics(yv, model.predict_proba(Xv)[:, 1], threshold,
                  "VALIDATION" if val_path else "HOLD-OUT (10% of train)")
    return model, meta


# --------------------------------------------------------------------------
# Load / predict / evaluate  (plug in any validation.csv here)
# --------------------------------------------------------------------------
def load_model(model_dir="model_dir"):
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(model_dir, MODEL_FILE))
    with open(os.path.join(model_dir, META_FILE)) as f:
        meta = json.load(f)
    return model, meta


def evaluate_file(data_path, model_dir="model_dir", threshold=0.5,
                  save_preds=None):
    model, meta = load_model(model_dir)
    X, y = transform(load_raw(data_path), meta)
    proba = model.predict_proba(X)[:, 1]
    metrics = print_metrics(y, proba, threshold, f"EVALUATION: {data_path}")
    if save_preds:
        pd.DataFrame({"proba": proba, "pred": (proba >= threshold).astype(int),
                      "actual": y}).to_csv(save_preds, index=False)
        print(f"Predictions saved to {save_preds}")
    return metrics


# --------------------------------------------------------------------------
# Edit these and run:  python xgb_attack.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    MODE = "validate"                      # "train" or "validate"

    TRAIN_PATH = "C:\\Users\\nitin\\Downloads\\shared-20261008T082342Z-1-001\\data\\train.csv"
    VAL_PATH = None                     # optional: validation CSV used for early stopping during train
    MODEL_DIR = "model_dir"
    THRESHOLD = 0.5

    # used when MODE == "validate"
    EVAL_PATH  = "C:\\Users\\nitin\\Downloads\\shared-20261008T082342Z-1-001\\data\\validation.csv"
    SAVE_PREDS = None                   # e.g. "preds.csv" or None

    if MODE == "train":
        train(TRAIN_PATH, VAL_PATH, MODEL_DIR, THRESHOLD)
    elif MODE == "validate":
        evaluate_file(EVAL_PATH, MODEL_DIR, THRESHOLD, SAVE_PREDS)
    else:
        raise ValueError("MODE must be 'train' or 'validate'")
