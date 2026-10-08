"""
Cost-sensitive, drift-robust XGBoost IDS  (weighted cross-entropy + DRO).

MODES (set MODE in the parameter block at the bottom; ALL parameters live there):
  train    - fit, pick threshold on a held-out calibration split, save model
  validate - score a labelled file (EVAL_PATH): metrics, per-family recall, drift-scenario table
  predict  - score an unlabelled file (PREDICT_PATH) -> submission CSV (row_id, prediction, probability)
  ablate   - drop feature groups one at a time, compare cost under simulated drift (decides DROP_FEATURES)
  lofo     - leave-one-family-out: train without family F, test on F. Gives the zero-day estimate,
             a LOFO-calibrated threshold (saved to lofo_results.json) and one scalar score to tune against.

DRO_METHOD
  "none"  - plain cost-weighted cross-entropy
  "group" - Group-DRO over attack families (multiplicative weights on group losses)
  "kl"    - KL-DRO: worst-case row reweighting  q_i ~ exp(loss_i / lambda), computed from cross-fitted
            (out-of-fold) losses, capped per row so mislabelled / conflicting rows cannot dominate
  "both"  - group weights x KL weights

LEAKAGE SAFEGUARDS
  * attack_cat is used ONLY as the Group-DRO group label, for per-family reports and for LOFO folds.
    Never a feature (it does not exist in unlabelled test files anyway).
  * train.csv is split by a hash of the feature row into  fit / hold-out / calibration.
    Identical rows always land in the same split, so duplicates cannot leak across splits.
  * Category lists are learned on the fit split only.
  * fit         -> trains the trees
    hold-out    -> early stopping + DRO losses + round selection
    calibration -> decision threshold (and ablation scoring) ONLY
    EVAL_PATH   -> reporting only. Never used for any choice.
  * LOFO: the held-out family is removed from fit / hold-out / calibration before anything is learned
    (category lists, early stopping, threshold). It is used only to score the fold.
"""
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             fbeta_score, precision_score, recall_score, roc_auc_score)

model_file = "model.json"
meta_file = "meta.json"
lofo_file = "lofo_results.json"


# --------------------------------------------------------------------------
# Loading / preprocessing
# --------------------------------------------------------------------------
def load_raw(path, feature_names=None, categorical_cols=()):
    """
    Headerless CSV. With feature_names: 38 columns = unlabelled, 40 = labelled (+ attack_cat, Label).
    Without feature_names: assumes labelled, generic names c0.. (used by leak_checks.py).
    Labelled frames get a 'group' (attack_cat, blank -> 'normal') and a 'label' column.
    """
    n = pd.read_csv(path, header=None, nrows=1).shape[1]
    if feature_names is None:
        names, labelled = [f"c{i}" for i in range(n - 2)], True
    else:
        k = len(feature_names)
        if n == k + 2:
            labelled = True
        elif n == k:
            labelled = False
        else:
            raise ValueError(f"{path}: {n} columns, expected {k} (unlabelled) or {k + 2} (labelled).")
        names = list(feature_names)
    all_names = names + (["group", "label"] if labelled else [])
    str_cols = set(categorical_cols) | {"group"}
    dtype = {c: str for c in all_names if c in str_cols}
    df = pd.read_csv(path, header=None, names=all_names, dtype=dtype, skipinitialspace=True)
    if labelled:
        g = df["group"].fillna("").astype(str).str.strip()
        df["group"] = g.where(~g.isin(["", "nan", "None"]), "normal")
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    return df


def _feature_cols(df):
    return [c for c in df.columns if c not in ("group", "label")]


def fit_preprocessor(df, categorical_cols=None, drop_cols=()):
    feats = [c for c in _feature_cols(df) if c not in set(drop_cols)]
    if categorical_cols:
        cat_cols = [c for c in feats if c in set(categorical_cols)]
    else:
        cat_cols = [c for c in feats if df[c].dtype == object]
    cats = {c: sorted(df[c].fillna("missing").astype(str).str.strip().unique().tolist()) for c in cat_cols}
    return {"columns": feats, "categories": cats, "drop": list(drop_cols)}


def transform(df, meta):
    missing = [c for c in meta["columns"] if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing expected feature columns: {missing[:5]}")
    X = df[meta["columns"]].copy()
    for c in meta["columns"]:
        if c in meta["categories"]:
            X[c] = pd.Categorical(X[c].fillna("missing").astype(str).str.strip(),
                                  categories=meta["categories"][c])
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    y = df["label"].astype(int).values if "label" in df.columns else None
    return X, y


def merge_rare_groups(groups, min_size):
    counts = groups.value_counts()
    return groups.where(~groups.isin(counts[counts < min_size].index), "rare_other")


def cost_weights(y, cost_fn, cost_fp):
    return np.where(y == 1, cost_fn, cost_fp).astype(float)


def _hash_u(df, seed):
    """Deterministic uniform [0,1) value per feature row. Identical rows get identical values."""
    cols = [c for c in df.columns if c not in ("group", "label")]
    h = pd.util.hash_pandas_object(df[cols], index=False).values.astype(np.uint64)
    x = h ^ np.uint64((seed * 0x9E3779B97F4A7C15 + 0x1234567) % (1 << 64))
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)).astype(np.float64) / 9007199254740992.0


def hash_split(df, holdout_frac, calib_frac, seed):
    """0 = fit, 1 = hold-out, 2 = calibration. Identical feature rows always share a split."""
    u = _hash_u(df, seed)
    split = np.zeros(len(df), dtype=int)
    split[u < holdout_frac] = 1
    split[(u >= holdout_frac) & (u < holdout_frac + calib_frac)] = 2
    return split


# --------------------------------------------------------------------------
# Drift simulation (latency -> timing columns, bandwidth -> load columns)
# --------------------------------------------------------------------------
def apply_shift(X, time_cols, rate_cols, time_scale, rate_scale):
    if np.isscalar(time_scale) and np.isscalar(rate_scale) and time_scale == 1 and rate_scale == 1:
        return X
    X = X.copy()
    for cols, s in ((time_cols, time_scale), (rate_cols, rate_scale)):
        for c in cols:
            if c in X.columns:
                X[c] = X[c].to_numpy(dtype=float) * s
    return X


def random_shift(X, cfg, rng, frac):
    """Per-row log-uniform rescaling of timing and load columns for a random `frac` of rows."""
    n, S = len(X), cfg["AUG_MAX_SCALE"]
    lo, hi = np.log(1.0 / S), np.log(S)
    mask_t, mask_r = rng.random_sample(n) < frac, rng.random_sample(n) < frac
    st = np.where(mask_t, np.exp(rng.uniform(lo, hi, n)), 1.0)
    sr = np.where(mask_r, np.exp(rng.uniform(lo, hi, n)), 1.0)
    return apply_shift(X, cfg["AUG_TIME_COLS"], cfg["AUG_RATE_COLS"], st, sr)


def scenario_probs(model, X, cfg):
    return {sc["name"]: model.predict_proba(
        apply_shift(X, cfg["AUG_TIME_COLS"], cfg["AUG_RATE_COLS"], sc["time"], sc["rate"]))[:, 1]
        for sc in cfg["SHIFT_SCENARIOS"]}


def scenario_report(model, X, y, thr, cfg, title):
    a, b = cfg["COST_FN"], cfg["COST_FP"]
    rows = []
    for name, p in scenario_probs(model, X, cfg).items():
        m = compute_metrics(y, p, thr, a, b)
        rows.append(dict(scenario=name, recall=m["recall"], precision=m["precision"], f1=m["f1"],
                         FN=m["FN"], FP=m["FP"], cost=m["total_cost"]))
    df = pd.DataFrame(rows).set_index("scenario")
    print(f"\n----- {title} (threshold={thr:.4f}) -----")
    print(df.to_string(float_format=lambda v: f"{v:.4f}"))
    return df


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def compute_metrics(y, proba, threshold, cost_fn, cost_fp):
    pred = (proba >= threshold).astype(int)
    fn = int(((y == 1) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    beta = float(np.sqrt(cost_fn / cost_fp))
    m = {"accuracy": accuracy_score(y, pred),
         "precision": precision_score(y, pred, zero_division=0),
         "recall": recall_score(y, pred, zero_division=0),
         "f1": f1_score(y, pred, zero_division=0),
         "fbeta": fbeta_score(y, pred, beta=beta, zero_division=0),
         "FN": fn, "FP": fp, "total_cost": cost_fn * fn + cost_fp * fp}
    if len(np.unique(y)) == 2:
        m["roc_auc"] = roc_auc_score(y, proba)
        m["pr_auc"] = average_precision_score(y, proba)
    m["confusion_matrix"] = confusion_matrix(y, pred, labels=[0, 1]).tolist()
    return m


def print_metrics(y, proba, threshold, cost_fn, cost_fp, title, groups=None):
    m = compute_metrics(y, proba, threshold, cost_fn, cost_fp)
    print(f"\n===== {title} =====")
    print(f"threshold={threshold:.4f}  cost = {cost_fn:g}*FN + {cost_fp:g}*FP  (fbeta beta={np.sqrt(cost_fn / cost_fp):.2f})")
    for k, v in m.items():
        if k != "confusion_matrix":
            print(f"{k:>12}: {v:.4f}" if isinstance(v, float) else f"{k:>12}: {v}")
    tn, fp, fn, tp = np.array(m["confusion_matrix"]).ravel()
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(f"            pred 0   pred 1\n  actual 0  {tn:7d}  {fp:7d}\n  actual 1  {fn:7d}  {tp:7d}")
    if groups is not None:
        pred = (proba >= threshold).astype(int)
        rep = (pd.DataFrame({"group": np.asarray(groups), "label": y, "flagged": pred})
               .groupby("group").agg(n=("label", "size"), label_rate=("label", "mean"),
                                     flagged_rate=("flagged", "mean"))
               .sort_values(["label_rate", "flagged_rate"], ascending=[False, True]))
        print("Per-group (attack groups: flagged_rate = recall; normal: flagged_rate = false-alarm rate):")
        print(rep.to_string(float_format=lambda v: f"{v:.4f}"))
    return m


# --------------------------------------------------------------------------
# Data preparation (shared by train / ablate / lofo)
# --------------------------------------------------------------------------
def prepare(cfg, df=None, holdout_family=None):
    """
    Builds fit / hold-out / calibration splits.
    holdout_family (LOFO): every row of that family is removed from all three splits before anything is
    learned; its attack rows are returned as S['X_test'], S['y_test'] (never seen in training).
    """
    if df is None:
        df = load_raw(cfg["TRAIN_PATH"], cfg["FEATURE_NAMES"], cfg["CATEGORICAL_COLS"])
    if "label" not in df.columns:
        raise ValueError("TRAIN_PATH must be a labelled file.")
    split = hash_split(df, cfg["HOLDOUT_FRAC"], cfg["CALIB_FRAC"], cfg["SEED"])

    d_test = None
    if holdout_family is not None:
        is_f = (df["group"] == holdout_family).to_numpy()
        d_test = df[is_f & (df["label"].to_numpy() == 1)].reset_index(drop=True)
        keep = ~is_f
        df, split = df[keep].reset_index(drop=True), split[keep]
        print(f"LOFO: held out '{holdout_family}' -> {len(d_test):,} unseen attack rows; "
              f"{int((~keep).sum()):,} rows removed from training")

    d_fit, d_hold, d_cal = [df[split == k].reset_index(drop=True) for k in range(3)]
    print(f"Split (hash of feature row): fit={len(d_fit):,}  hold-out={len(d_hold):,}  calibration={len(d_cal):,}")

    meta = fit_preprocessor(d_fit, cfg["CATEGORICAL_COLS"], cfg["DROP_FEATURES"])   # fit split only
    X_fit, y_fit = transform(d_fit, meta)
    X_hold, y_hold = transform(d_hold, meta)
    X_cal, y_cal = transform(d_cal, meta)

    g_fit = merge_rare_groups(d_fit["group"], cfg["MIN_GROUP_SIZE"])
    known = set(g_fit.unique())
    g_hold = d_hold["group"].where(d_hold["group"].isin(known), "rare_other")
    g_cal = d_cal["group"].where(d_cal["group"].isin(known), "rare_other")

    # cross-fitting folds for KL-DRO (hash of feature row -> duplicates stay in one fold)
    kf = int(cfg["KL_OOF_FOLDS"])
    fold_fit = (np.minimum((_hash_u(d_fit, cfg["SEED"] + 1) * kf).astype(int), kf - 1)
                if kf >= 2 else np.zeros(len(d_fit), dtype=int))

    rng = np.random.RandomState(cfg["SEED"])
    k = min(100_000, len(y_fit))
    idx = rng.choice(len(y_fit), k, replace=False)
    S = dict(meta=meta, X_fit_s=X_fit.iloc[idx], y_fit_s=y_fit[idx])         # clean sample for train report

    if cfg["AUG_FRAC"] > 0:
        X_fit = random_shift(X_fit, cfg, rng, cfg["AUG_FRAC"])
    if cfg["EVAL_INCLUDE_SHIFTED"]:
        X_hold = pd.concat([X_hold, random_shift(X_hold, cfg, rng, 1.0)], ignore_index=True)
        y_hold = np.r_[y_hold, y_hold]
        g_hold = pd.concat([g_hold, g_hold], ignore_index=True)
    S.update(X_fit=X_fit, y_fit=y_fit, g_fit=g_fit, fold_fit=fold_fit,
             X_hold=X_hold, y_hold=y_hold, g_hold=g_hold,
             X_cal=X_cal, y_cal=y_cal, g_cal=g_cal)
    if d_test is not None:
        S["X_test"], S["y_test"] = transform(d_test, meta)
    return S


# --------------------------------------------------------------------------
# Training: weighted cross-entropy + Group-DRO / KL-DRO
# --------------------------------------------------------------------------
def _fit(X_tr, y_tr, w_tr, X_ev, y_ev, w_ev, params):
    model = xgb.XGBClassifier(enable_categorical=True, **params)
    model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_ev, y_ev)],
              sample_weight_eval_set=[w_ev], verbose=False)
    return model


def _row_losses(p, y, cost_fn, cost_fp):
    """Per-row cost-weighted log-loss."""
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(cost_fn * y * np.log(p) + cost_fp * (1 - y) * np.log(1 - p))


# ---- KL-DRO helpers ----
def fit_row_losses(model, S, cfg, w_tr):
    """
    Per-row losses on the fit rows. With KL_OOF_FOLDS >= 2 these are cross-fitted (out-of-fold), so rows the
    model has memorised still show their true difficulty. With 0 they are in-sample (cheap, but depth-8
    trees drive most training losses to ~0).
    """
    a, b = cfg["COST_FN"], cfg["COST_FP"]
    X, y, kf = S["X_fit"], S["y_fit"], int(cfg["KL_OOF_FOLDS"])
    if kf < 2:
        return _row_losses(model.predict_proba(X)[:, 1], y, a, b)
    n_trees = int(min(model.best_iteration + 1, cfg["KL_OOF_MAX_TREES"]))
    params = {k_: v for k_, v in cfg["XGB_PARAMS"].items() if k_ != "early_stopping_rounds"}
    params["n_estimators"] = n_trees
    p = np.empty(len(y))
    for f in range(kf):
        tr, te = np.flatnonzero(S["fold_fit"] != f), np.flatnonzero(S["fold_fit"] == f)
        m = xgb.XGBClassifier(enable_categorical=True, **params)
        m.fit(X.iloc[tr], y[tr], sample_weight=w_tr[tr])
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
    return _row_losses(p, y, a, b)


def solve_kl_lambda(l, target_ess):
    """
    Temperature lambda such that weights exp(l/lambda) have effective sample size = target_ess * N
    (before capping). Scale-free replacement for choosing the KL radius directly.
    """
    l = np.asarray(l, dtype=float)
    if len(l) > 200_000:
        l = l[::len(l) // 200_000 + 1]
    s = l.std() + 1e-12
    lo, hi = np.log(s * 1e-3), np.log(s * 1e3)

    def ess(lam):
        w = np.exp((l - l.max()) / lam)
        return (w.sum() ** 2) / (len(w) * (w ** 2).sum())

    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if ess(np.exp(mid)) < target_ess:
            lo = mid
        else:
            hi = mid
    return float(np.exp(hi))


def kl_weights(l, cfg, lam):
    """Worst-case reweighting q_i ~ exp(l_i / lam), capped, mixed toward uniform, mean 1."""
    w = np.exp((l - l.max()) / lam)
    w = w / w.mean()
    w = np.minimum(w, cfg["KL_MAX_WEIGHT"])                   # protects against mislabelled rows
    mix = cfg["KL_PRIOR_MIX"]
    w = (1 - mix) * w + mix
    return w / w.mean()


def run_dro(S, cfg):
    a, b = cfg["COST_FN"], cfg["COST_FP"]
    method = cfg["DRO_METHOD"]
    if method not in ("none", "group", "kl", "both"):
        raise ValueError(f"DRO_METHOD must be none|group|kl|both, got {method!r}")
    use_g, use_kl = method in ("group", "both"), method in ("kl", "both")

    g_tr, y_tr = S["g_fit"], S["y_fit"]
    glist = sorted(g_tr.unique())
    n_g, N = g_tr.value_counts(), len(y_tr)
    prior = (n_g / N).reindex(glist)
    q = prior.copy()                                   # round 0 == plain weighted cross-entropy
    kw, lam = np.ones(N), None
    c_tr, c_ho = cost_weights(y_tr, a, b), cost_weights(S["y_hold"], a, b)
    g_hold = np.asarray(S["g_hold"])
    rounds = cfg["DRO_ROUNDS"] if method != "none" else 1
    print(f"\nDRO_METHOD={method}   Groups ({len(glist)}): " + ", ".join(f"{g}={n_g[g]}" for g in glist))

    best = dict(model=None, crit=np.inf, q=None, round=-1, lam=None)
    for r in range(rounds):
        if use_g:
            gw = q.reindex(g_tr.values).values * N / n_g.reindex(g_tr.values).values
            gw = np.minimum(gw, cfg["DRO_MAX_WEIGHT"])
        else:
            gw = np.ones(N)
        w_tr = c_tr * gw * kw
        model = _fit(S["X_fit"], y_tr, w_tr, S["X_hold"], S["y_hold"], c_ho, cfg["XGB_PARAMS"])

        l_ho = _row_losses(model.predict_proba(S["X_hold"])[:, 1], S["y_hold"], a, b)
        L = pd.Series(l_ho).groupby(g_hold).mean()
        crit = float(L.max())                                           # worst-group hold-out loss
        msg = (f"[round {r}] trees={model.best_iteration + 1:4d}  worst group='{L.idxmax()}' "
               f"loss={L.max():.4f}  mean group loss={L.mean():.4f}")

        l_fit = None
        if use_kl and (lam is None or r < rounds - 1):
            l_fit = fit_row_losses(model, S, cfg, w_tr)
            if lam is None:
                lam = solve_kl_lambda(l_fit, cfg["KL_TARGET_ESS"])      # fixed from round 0 onward
        if use_kl:
            R = float(np.mean(kl_weights(l_ho, cfg, lam) * l_ho))      # loss under the capped worst case
            msg += f"  KL-robust hold-out loss={R:.4f}"
            if method == "kl":
                crit = R
        print(msg)

        if crit < best["crit"]:
            best = dict(model=model, crit=crit, q=q.copy(), round=r, lam=lam)

        if use_g:
            Lq = L.reindex(glist).fillna(L.mean())
            q = q * np.exp(cfg["DRO_STEP_SIZE"] * Lq.values)
            q = q / q.sum()
            q = (1 - cfg["DRO_PRIOR_MIX"]) * q + cfg["DRO_PRIOR_MIX"] * prior
        if use_kl and r < rounds - 1:
            kw = kl_weights(l_fit, cfg, lam)
            ess = kw.sum() ** 2 / (N * (kw ** 2).sum())
            per_g = pd.Series(kw).groupby(g_tr.values).mean()
            print(f"    KL weights: lambda={lam:.4g}  ESS={ess:.2f}N  max={kw.max():.2f}  "
                  f"rows with w>2: {(kw > 2).mean():.2%}  attack weight share={kw[y_tr == 1].sum() / kw.sum():.2%} "
                  f"(raw {y_tr.mean():.2%})")
            print("    mean KL weight per group: " + ", ".join(f"{g}={v:.2f}" for g, v in per_g.sort_values(ascending=False).items()))

    print(f"\nKept round {best['round']} (criterion {best['crit']:.4f}, "
          f"{'KL-robust' if method == 'kl' else 'worst-group'} hold-out loss)")
    if use_g:
        print("Group weights q (kept round):")
        print(best["q"].sort_values(ascending=False).to_string(float_format=lambda v: f"{v:.4f}"))
    return best


def pick_threshold(model, S, cfg):
    """Scan thresholds on the calibration split (clean + drift scenarios). Ties -> lower threshold."""
    mode = cfg["THRESHOLD_MODE"]
    if mode == "fixed":
        return float(cfg["THRESHOLD"])
    if mode == "lofo":
        path = os.path.join(cfg["MODEL_DIR"], lofo_file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"THRESHOLD_MODE='lofo' needs {path}; run MODE='lofo' first.")
        with open(path) as f:
            t = float(json.load(f)["lofo_threshold"])
        print(f"\nThreshold taken from LOFO run: {t:.4f}")
        return t
    a, b = cfg["COST_FN"], cfg["COST_FP"]
    y = S["y_cal"]
    pos, neg = y == 1, y == 0
    probs = scenario_probs(model, S["X_cal"], cfg)
    grid = np.linspace(cfg["THRESHOLD_GRID_MIN"], cfg["THRESHOLD_GRID_MAX"], cfg["THRESHOLD_GRID_STEPS"])
    agg_fn = max if cfg["THRESHOLD_OBJECTIVE"] == "worst" else (lambda v: float(np.mean(v)))
    best_t, best_v = 0.5, np.inf
    for t in grid:
        costs = [a * (pos & (p < t)).sum() + b * (neg & (p >= t)).sum() for p in probs.values()]
        v = agg_fn(costs)
        if v < best_v - 1e-9:
            best_t, best_v = float(t), v
    costs05 = [a * (pos & (p < 0.5)).sum() + b * (neg & (p >= 0.5)).sum() for p in probs.values()]
    print(f"\nThreshold scan on calibration split ({cfg['THRESHOLD_OBJECTIVE']} cost over "
          f"{len(probs)} scenarios): chosen {best_t:.4f} -> {best_v:,.0f}   (at 0.5 -> {agg_fn(costs05):,.0f})")
    return best_t


def train(cfg):
    a, b = cfg["COST_FN"], cfg["COST_FP"]
    os.makedirs(cfg["MODEL_DIR"], exist_ok=True)
    S = prepare(cfg)
    best = run_dro(S, cfg)
    model = best["model"]
    thr = pick_threshold(model, S, cfg)

    model.save_model(os.path.join(cfg["MODEL_DIR"], model_file))
    meta = S["meta"]
    meta.update({"threshold": thr, "cost_fn": a, "cost_fp": b, "best_round": best["round"],
                 "dro_method": cfg["DRO_METHOD"], "kl_lambda": best["lam"],
                 "group_weights": best["q"].to_dict()})
    with open(os.path.join(cfg["MODEL_DIR"], meta_file), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved model + metadata to '{cfg['MODEL_DIR']}'  (features used: {len(meta['columns'])})")

    print_metrics(S["y_fit_s"], model.predict_proba(S["X_fit_s"])[:, 1], thr, a, b, "TRAIN SAMPLE (clean)")
    print_metrics(S["y_cal"], model.predict_proba(S["X_cal"])[:, 1], thr, a, b,
                  "CALIBRATION SPLIT (clean; threshold was tuned here, so slightly optimistic)", groups=S["g_cal"])
    scenario_report(model, S["X_cal"], S["y_cal"], thr, cfg, "CALIBRATION SPLIT under simulated drift")
    return model, meta


# --------------------------------------------------------------------------
# LOFO: leave-one-family-out (zero-day estimate)
# --------------------------------------------------------------------------
def lofo(cfg):
    """
    For each attack family F: train without F (fit / hold-out / calibration all exclude F), then score
      * all attack rows of F          (unseen family  -> zero-day recall)
      * the calibration-split normals (-> false alarms)
    Pooled cost = a * (sum of FN over families) + b * (mean over folds of FP), so normals are counted once.
    The LOFO threshold minimises that pooled cost (worst or mean over drift scenarios, as THRESHOLD_OBJECTIVE).
    Note: that threshold is picked and reported on the same scores (one parameter, so mild optimism), and
    fold models are trained on slightly less data than the final model.
    """
    a, b = cfg["COST_FN"], cfg["COST_FP"]
    fcfg = {**cfg, **cfg["LOFO_OVERRIDES"], "THRESHOLD_MODE": "scan"}
    scen = [s["name"] for s in fcfg["SHIFT_SCENARIOS"]]
    clean = "clean" if "clean" in scen else scen[0]

    df = load_raw(cfg["TRAIN_PATH"], cfg["FEATURE_NAMES"], cfg["CATEGORICAL_COLS"])
    df["group"] = merge_rare_groups(df["group"], cfg["MIN_GROUP_SIZE"])
    att = df.loc[df["label"] == 1, "group"].value_counts()
    att = att[att.index != "normal"]
    families = list(cfg["LOFO_FAMILIES"]) if cfg["LOFO_FAMILIES"] else list(att.index)
    print(f"\nLOFO families (attack rows): " + ", ".join(f"{f}={att.get(f, 0):,}" for f in families))
    if len(families) < 2:
        raise ValueError("LOFO needs at least two attack families.")

    results = {}
    for F in families:
        print(f"\n{'=' * 78}\nLOFO fold: hold out '{F}'\n{'=' * 78}")
        S = prepare(fcfg, df=df, holdout_family=F)
        best = run_dro(S, fcfg)
        model = best["model"]
        thr_in = pick_threshold(model, S, fcfg)                  # chosen on SEEN families only
        pos = scenario_probs(model, S["X_test"], fcfg)
        neg_mask = S["y_cal"] == 0
        neg = scenario_probs(model, S["X_cal"][neg_mask], fcfg)
        results[F] = dict(pos={k: np.sort(v) for k, v in pos.items()},
                          neg={k: np.sort(v) for k, v in neg.items()},
                          thr_in=thr_in, n_pos=int(len(S["y_test"])), n_neg=int(neg_mask.sum()))
        r = results[F]
        rec = 1 - np.searchsorted(r["pos"][clean], thr_in, side="left") / r["n_pos"]
        print(f"-> '{F}': in-family threshold {thr_in:.4f} gives unseen-family recall {rec:.4f} "
              f"({r['n_pos']:,} attacks)")
        del S, model, best

    # ---- pooled threshold search (sorted scores + searchsorted: fast) ----
    grid = np.linspace(cfg["THRESHOLD_GRID_MIN"], cfg["THRESHOLD_GRID_MAX"], cfg["THRESHOLD_GRID_STEPS"])
    total_pos = sum(r["n_pos"] for r in results.values())

    def pooled_counts(thrs):
        thrs = np.atleast_1d(np.asarray(thrs, dtype=float))
        FN, FP = np.zeros((len(scen), len(thrs))), np.zeros((len(scen), len(thrs)))
        for i, sn in enumerate(scen):
            for r in results.values():
                FN[i] += np.searchsorted(r["pos"][sn], thrs, side="left")                     # p < t  -> missed
                FP[i] += r["neg"][sn].size - np.searchsorted(r["neg"][sn], thrs, side="left")  # p >= t -> alarm
            FP[i] /= len(results)
        return FN, FP

    agg = (lambda c: c.max(axis=0)) if cfg["THRESHOLD_OBJECTIVE"] == "worst" else (lambda c: c.mean(axis=0))
    FNg, FPg = pooled_counts(grid)
    obj = agg(a * FNg + b * FPg)
    t_lofo = float(grid[int(np.argmin(obj))])                    # argmin -> first (lowest) threshold on ties

    # ---- report ----
    thr_in = {F: results[F]["thr_in"] for F in families}
    cands = {"0.5": 0.5, "mean in-family thr": float(np.mean(list(thr_in.values())))}
    meta_path = os.path.join(cfg["MODEL_DIR"], meta_file)
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            cands["saved train thr"] = float(json.load(f)["threshold"])
    cands["LOFO thr"] = t_lofo

    rows = []
    for name, t in cands.items():
        FN, FP = pooled_counts([t])
        cost = a * FN[:, 0] + b * FP[:, 0]
        rows.append(dict(threshold_rule=name, thr=t, zero_day_recall=1 - FN[scen.index(clean), 0] / total_pos,
                         FN=FN[scen.index(clean), 0], FP=FP[scen.index(clean), 0],
                         cost_clean=cost[scen.index(clean)], cost_worst=cost.max(), cost_mean=cost.mean()))
    summary = pd.DataFrame(rows).set_index("threshold_rule")
    print(f"\n{'=' * 78}\nLOFO POOLED RESULT  (cost = {a:g}*FN + {b:g}*FP; FP = mean over folds)\n{'=' * 78}")
    print(summary.to_string(float_format=lambda v: f"{v:,.4f}" if abs(v) < 10 else f"{v:,.0f}"))

    fam_rows = []
    for F in families:
        r = results[F]
        row = dict(family=F, n_attacks=r["n_pos"], thr_in_family=r["thr_in"])
        for name, t in cands.items():
            row[f"recall@{name}"] = 1 - np.searchsorted(r["pos"][clean], t, side="left") / r["n_pos"]
        row["recall@own in-family thr"] = 1 - np.searchsorted(r["pos"][clean], r["thr_in"], side="left") / r["n_pos"]
        fam_rows.append(row)
    fam = pd.DataFrame(fam_rows).set_index("family")
    print("\nPer-family unseen recall (clean scenario):")
    print(fam.to_string(float_format=lambda v: f"{v:,.4f}" if abs(v) < 10 else f"{v:,.0f}"))

    FN, FP = pooled_counts([t_lofo])
    sc_tab = pd.DataFrame({"zero_day_recall": 1 - FN[:, 0] / total_pos, "FN": FN[:, 0], "FP": FP[:, 0],
                           "cost": a * FN[:, 0] + b * FP[:, 0]}, index=scen)
    print(f"\nPooled result under drift scenarios at LOFO threshold {t_lofo:.4f}:")
    print(sc_tab.to_string(float_format=lambda v: f"{v:,.4f}" if abs(v) < 10 else f"{v:,.0f}"))

    score = float(sc_tab["cost"].max() if cfg["THRESHOLD_OBJECTIVE"] == "worst" else sc_tab["cost"].mean())
    print(f"\n*** LOFO SCORE (tune against this; lower = better): {score:,.1f}  "
          f"[{cfg['THRESHOLD_OBJECTIVE']} scenario cost at LOFO threshold {t_lofo:.4f}] ***")
    print(f"*** Zero-day recall (clean) at LOFO threshold: {sc_tab.loc[clean, 'zero_day_recall']:.4f} ***")

    os.makedirs(cfg["MODEL_DIR"], exist_ok=True)
    out = dict(lofo_threshold=t_lofo, lofo_score=score, objective=cfg["THRESHOLD_OBJECTIVE"],
               dro_method=cfg["DRO_METHOD"], families=families,
               zero_day_recall_clean=float(sc_tab.loc[clean, "zero_day_recall"]),
               in_family_thresholds={k: float(v) for k, v in thr_in.items()},
               per_family_recall_at_lofo_thr={F: float(fam.loc[F, "recall@LOFO thr"]) for F in families})
    with open(os.path.join(cfg["MODEL_DIR"], lofo_file), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {os.path.join(cfg['MODEL_DIR'], lofo_file)}  (use THRESHOLD_MODE='lofo' in MODE='train' to adopt it)")
    return out


# --------------------------------------------------------------------------
# Ablation: which feature groups hurt robustness under drift?
# --------------------------------------------------------------------------
def ablate(cfg):
    a, b = cfg["COST_FN"], cfg["COST_FP"]
    S = prepare(cfg)
    params = {**cfg["XGB_PARAMS"], **cfg["ABLATION_XGB"]}
    c_fit, c_hold = cost_weights(S["y_fit"], a, b), cost_weights(S["y_hold"], a, b)
    variants = [("(all features)", [])] + [(f"- {n}", c) for n, c in cfg["FEATURE_GROUPS"].items()]
    rows = []
    for name, cols in variants:
        drop = [c for c in cols if c in S["X_fit"].columns]
        Xf, Xh, Xc = (S[k].drop(columns=drop) for k in ("X_fit", "X_hold", "X_cal"))
        model = _fit(Xf, S["y_fit"], c_fit, Xh, S["y_hold"], c_hold, params)
        costs, recs = [], []
        for p in scenario_probs(model, Xc, cfg).values():
            m = compute_metrics(S["y_cal"], p, 0.5, a, b)
            costs.append(m["total_cost"]); recs.append(m["recall"])
        rows.append(dict(variant=name, n_dropped=len(drop), clean_cost=costs[0], worst_cost=max(costs),
                         mean_cost=float(np.mean(costs)), worst_recall=min(recs)))
        print(f"  done: {name}")
    df = pd.DataFrame(rows).sort_values("worst_cost").set_index("variant")
    base = df.loc["(all features)", "worst_cost"]
    print(f"\n===== ABLATION on calibration split (cost = {a:g}*FN + {b:g}*FP, threshold 0.5, lower = better) =====")
    print(df.to_string(float_format=lambda v: f"{v:,.4f}" if v < 10 else f"{v:,.0f}"))
    better = df[df["worst_cost"] < base].index.tolist()
    better = [v for v in better if v != "(all features)"]
    print(f"\nVariants with lower worst-case cost than all features: {better if better else 'none'}")
    print("-> add the columns of those groups to DROP_FEATURES, then retrain. Differences within ~2-3% are noise.")
    return df


# --------------------------------------------------------------------------
# Load / validate / predict
# --------------------------------------------------------------------------
def load_model(model_dir):
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(model_dir, model_file))
    with open(os.path.join(model_dir, meta_file)) as f:
        meta = json.load(f)
    return model, meta


def evaluate_file(cfg):
    """Reporting only. Nothing here is used to tune anything."""
    model, meta = load_model(cfg["MODEL_DIR"])
    df = load_raw(cfg["EVAL_PATH"], cfg["FEATURE_NAMES"], cfg["CATEGORICAL_COLS"])
    if "label" not in df.columns:
        raise ValueError("EVAL_PATH has no labels. Use MODE='predict' for unlabelled files.")
    X, y = transform(df, meta)
    proba = model.predict_proba(X)[:, 1]
    thr = cfg["EVAL_THRESHOLD"] if cfg["EVAL_THRESHOLD"] is not None else meta["threshold"]
    a, b = cfg["COST_FN"], cfg["COST_FP"]
    m = print_metrics(y, proba, thr, a, b, f"EVALUATION: {cfg['EVAL_PATH']}", groups=df["group"])
    scenario_report(model, X, y, thr, cfg, "EVALUATION under simulated drift")
    if cfg["SAVE_PREDS"]:
        pd.DataFrame({"proba": proba, "pred": (proba >= thr).astype(int), "actual": y,
                      "group": df["group"].values}).to_csv(cfg["SAVE_PREDS"], index=False)
        print(f"Predictions saved to {cfg['SAVE_PREDS']}")
    return m


def predict_file(cfg):
    model, meta = load_model(cfg["MODEL_DIR"])
    df = load_raw(cfg["PREDICT_PATH"], cfg["FEATURE_NAMES"], cfg["CATEGORICAL_COLS"])
    X, _ = transform(df, meta)
    proba = model.predict_proba(X)[:, 1]
    thr = cfg["PREDICT_THRESHOLD"] if cfg["PREDICT_THRESHOLD"] is not None else meta["threshold"]
    sub = pd.DataFrame({"row_id": np.arange(1, len(proba) + 1),
                        "prediction": (proba >= thr).astype(int),
                        "probability": np.round(proba, 6)})
    sub.to_csv(cfg["SUBMISSION_PATH"], index=False)
    print(f"Threshold {thr:.4f} | flagged {sub['prediction'].mean():.2%} of {len(sub):,} rows")
    print(f"Saved submission to {cfg['SUBMISSION_PATH']}\n{sub.head().to_string(index=False)}")
    return sub


# ==========================================================================
# ALL PARAMETERS - EDIT HERE  (every UPPERCASE name below is passed to the code)
# ==========================================================================
if __name__ == "__main__":
    # ---- what to run ----------------------------------------------------
    MODE = "validate"                        # "train" | "validate" | "predict" | "ablate" | "lofo"

    # ---- paths ----------------------------------------------------------
    DATA_DIR = r"C:\Users\nitin\Downloads\shared-20261008T082342Z-1-001\data"
    TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
    EVAL_PATH = os.path.join(DATA_DIR, "validation.csv")       # labelled; used by MODE="validate"
    PREDICT_PATH = os.path.join(DATA_DIR, "test.csv")          # unlabelled; used by MODE="predict"
    SUBMISSION_PATH = os.path.join(DATA_DIR, "submission.csv")
    MODEL_DIR = os.path.join(DATA_DIR, "model_dir")
    SAVE_PREDS = None                    # e.g. os.path.join(DATA_DIR, "val_preds.csv")
    SEED = 42

    # ---- schema (from the organizers' baseline) -------------------------
    FEATURE_NAMES = [
        "proto", "state", "dur", "sbytes", "dbytes", "sloss", "dloss", "service",
        "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb",
        "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "Sjit", "Djit",
        "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports",
        "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst",
        "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm"]
    CATEGORICAL_COLS = ["proto", "state", "service"]
    DROP_FEATURES = []                   # columns excluded from the model; decide with MODE="ablate",
                                         # e.g. ["stcpb", "dtcpb"]

    # ---- cost-sensitive loss (weighted cross-entropy) -------------------
    COST_FN = 40.0                       # a: weight on attack rows. Hackathon cost = 20*FN + FP
    COST_FP = 1.0                        # b: weight on normal rows

    # ---- splits of train.csv (hash-based, duplicates stay together) ------
    HOLDOUT_FRAC = 0.10                  # early stopping + DRO losses
    CALIB_FRAC = 0.10                    # threshold selection + ablation scoring (never trained on)

    # ---- decision threshold ---------------------------------------------
    THRESHOLD_MODE = "scan"              # "scan" = choose on calibration split | "fixed" = use THRESHOLD
                                         # | "lofo" = use lofo_threshold from the last MODE="lofo" run
    THRESHOLD = 0.5                      # used when THRESHOLD_MODE == "fixed"
    THRESHOLD_OBJECTIVE = "worst"        # "worst" = min of worst-scenario cost | "mean" = mean over scenarios
    THRESHOLD_GRID_MIN = 0.02
    THRESHOLD_GRID_MAX = 0.95
    THRESHOLD_GRID_STEPS = 200
    EVAL_THRESHOLD = None                # MODE="validate": None = use saved threshold
    PREDICT_THRESHOLD = None             # MODE="predict":  None = use saved threshold

    # ---- DRO ------------------------------------------------------------
    DRO_METHOD = "both"                    # "none" | "group" | "kl" | "both"
    DRO_ROUNDS = 4                       # retrain rounds; keeps the round with the best hold-out criterion
    MIN_GROUP_SIZE = 500                 # smaller groups are merged into "rare_other"
    # Group-DRO (groups = attack families)
    DRO_STEP_SIZE = 0.5                  # eta in q_g <- q_g * exp(eta * loss_g)
    DRO_PRIOR_MIX = 0.3                  # pull q back toward empirical group frequencies (0 = none)
    DRO_MAX_WEIGHT = 10.0                # cap on a group's per-row weight multiplier
    # KL-DRO (row level): w_i ~ exp(cost-weighted loss_i / lambda), lambda set by the target ESS
    KL_TARGET_ESS = 0.5                  # effective sample size / N before capping. Lower = more adversarial
    KL_MAX_WEIGHT = 5.0                  # per-row weight cap (guards against mislabelled / conflicting rows)
    KL_PRIOR_MIX = 0.3                   # mix toward uniform weights (0 = none)
    KL_OOF_FOLDS = 2                     # cross-fitted losses for the weights; 0 = in-sample (cheap, weaker)
    KL_OOF_MAX_TREES = 300               # trees per cross-fit model

    # ---- LOFO (MODE="lofo") ---------------------------------------------
    LOFO_FAMILIES = None                 # None = every attack family (after merging rare ones), or a list
    LOFO_OVERRIDES = dict(DRO_ROUNDS=3)  # cfg overrides for fold training only (keeps LOFO affordable)

    # ---- drift robustness -----------------------------------------------
    AUG_FRAC = 0.5                       # fraction of fit rows randomly rescaled during training (0 = off)
    AUG_MAX_SCALE = 3.0                  # random factor drawn log-uniformly from [1/S, S]
    AUG_TIME_COLS = ["dur", "Sjit", "Djit", "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat"]   # latency-sensitive
    AUG_RATE_COLS = ["Sload", "Dload"]   # bandwidth-sensitive
    EVAL_INCLUDE_SHIFTED = True          # hold-out also contains a randomly shifted copy (early stop / DRO)
    SHIFT_SCENARIOS = [                  # fixed stress tests: threshold scan, ablation, validate report
        dict(name="clean",            time=1.0, rate=1.0),
        dict(name="latency x3",       time=3.0, rate=1.0),
        dict(name="latency x0.4",     time=0.4, rate=1.0),
        dict(name="bandwidth x0.3",   time=1.0, rate=0.3),
        dict(name="bandwidth x3",     time=1.0, rate=3.0),
        dict(name="congestion",       time=3.0, rate=0.33),
    ]

    # ---- ablation (MODE="ablate") ---------------------------------------
    FEATURE_GROUPS = {
        "tcp_seq_numbers": ["stcpb", "dtcpb"],
        "latency_timing":  AUG_TIME_COLS,
        "load_rates":      AUG_RATE_COLS,
        "tcp_windows":     ["swin", "dwin"],
        "packet_loss":     ["sloss", "dloss"],
        "ct_counters":     ["ct_flw_http_mthd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
                            "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm"],
        "app_layer":       ["trans_depth", "res_bdy_len", "is_ftp_login", "ct_ftp_cmd"],
    }
    ABLATION_XGB = dict(n_estimators=300, learning_rate=0.15, max_depth=6, early_stopping_rounds=20)

    # ---- XGBoost --------------------------------------------------------
    XGB_PARAMS = dict(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        gamma=0.0,
        tree_method="hist",
        eval_metric="logloss",           # cost-weighted on the hold-out for early stopping
        early_stopping_rounds=50,
        random_state=SEED,
        n_jobs=-1,
    )
    # ======================================================================

    cfg = {k: v for k, v in globals().items() if k.isupper()}
    print("CONFIG")
    for k, v in cfg.items():
        print(f"  {k:<22} = {v}")

    {"train": train, "validate": evaluate_file, "predict": predict_file,
     "ablate": ablate, "lofo": lofo}[MODE](cfg)
