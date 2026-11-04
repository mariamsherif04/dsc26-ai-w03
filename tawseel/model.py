import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "failed"
# Known at creation time. Dropped: parcel_id, courier_notes, delivered_at, failed.
# customer_id/created_at are kept only so history features can be computed;
# the pipeline does not use customer_id as a feature.
FEATURE_COLUMNS = [
    "customer_id", "merchant_id", "area", "created_at",
    "promised_slot", "cod_amount", "n_items", "address_text",
]
HISTORY_COLUMNS = ["customer_prior_parcels", "customer_prior_fail_rate"]
CAT_COLS = ["area", "merchant_id", "promised_slot"]
NUM_COLS = ["cod_log", "n_items", "hour", "dow",
            "addr_empty", "addr_len", "addr_arabic", "addr_digits",
            "customer_prior_parcels", "customer_prior_fail_rate"]


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    keys = ["customer_id", "created_at"]
    d = df[keys + [TARGET]].copy()
    g = (d.groupby(keys, dropna=False)[TARGET]
          .agg(n="size", f="sum").reset_index()
          .sort_values(keys, kind="stable"))
    # cumulative totals *before* each (customer, timestamp) group -> ties excluded
    g["prior_n"] = g.groupby("customer_id", dropna=False)["n"].cumsum() - g["n"]
    g["prior_f"] = g.groupby("customer_id", dropna=False)["f"].cumsum() - g["f"]
    m = d[keys].merge(g[keys + ["prior_n", "prior_f"]], on=keys, how="left")
    n = m["prior_n"].to_numpy()
    f = m["prior_f"].to_numpy(dtype=float)
    out["customer_prior_parcels"] = n.astype(int)
    out["customer_prior_fail_rate"] = np.where(n > 0, f / np.maximum(n, 1), np.nan)
    return out


def time_split(df, time_col="created_at", test_fraction=0.2):
    if not (0 < test_fraction < 1):
        raise ValueError("test_fraction must be in (0, 1)")
    n_test = int(np.ceil(len(df) * test_fraction))
    srt = df[time_col].sort_values(kind="stable")
    boundary = srt.iloc[len(df) - n_test]
    return df[df[time_col] < boundary], df[df[time_col] >= boundary]


def _derive(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    cod = pd.to_numeric(df["cod_amount"], errors="coerce").clip(lower=0)
    X["cod_log"] = np.log1p(cod)
    X["n_items"] = pd.to_numeric(df["n_items"], errors="coerce")
    ts = pd.to_datetime(df["created_at"], errors="coerce")
    X["hour"] = ts.dt.hour.astype(float)
    X["dow"] = ts.dt.dayofweek.astype(float)
    addr = df["address_text"].astype(object).where(df["address_text"].notna(), "").astype(str)
    X["addr_empty"] = (addr.str.strip() == "").astype(float)
    X["addr_len"] = np.log1p(addr.str.len().astype(float))
    X["addr_arabic"] = addr.str.count(r"[\u0600-\u06FF]").astype(float)
    X["addr_digits"] = addr.str.count(r"\d").astype(float)
    for c in HISTORY_COLUMNS:
        X[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    for c in CAT_COLS:
        s = df[c]
        X[c] = s.astype(object).where(s.notna(), "missing").astype(str)
    return X


def build_pipeline(seed: int = 0) -> Pipeline:
    num = Pipeline([
        ("imp", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("sc", StandardScaler()),
    ])
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20)
    prep = ColumnTransformer([("num", num, NUM_COLS), ("cat", cat, CAT_COLS)],
                             remainder="drop")
    return Pipeline([
        ("derive", FunctionTransformer(_derive)),
        ("prep", prep),
        ("clf", LogisticRegression(C=0.5, max_iter=2000, random_state=seed)),
    ])


def evaluate(model, test_df: pd.DataFrame) -> dict:
    y = test_df[TARGET]
    p = model.predict_proba(test_df)[:, 1]
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "base_rate": float(y.mean()),
        "n": int(len(test_df)),
    }


def compare_to_baseline(train_df, test_df, seed: int = 0) -> dict:
    model = build_pipeline(seed).fit(train_df, train_df[TARGET])
    dummy = DummyClassifier(strategy="prior").fit(train_df, train_df[TARGET])
    m, d = evaluate(model, test_df), evaluate(dummy, test_df)
    return {"model": m, "dummy": d, "lift_pr_auc": m["pr_auc"] - d["pr_auc"]}