# /data/weizhen/code/math/eval_wrmsse_from_submission.py
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path("/data/weizhen/code/math")
SUB_PATH = BASE_DIR / "submission.csv"
TR_LAST = 1913
H = 28

HIER_LIST = [
    [],                                # 1: all
    ["state_id"],                      # 2
    ["store_id"],                      # 3
    ["cat_id"],                        # 4
    ["dept_id"],                       # 5
    ["state_id", "cat_id"],            # 6
    ["state_id", "dept_id"],           # 7
    ["store_id", "cat_id"],            # 8
    ["store_id", "dept_id"],           # 9
    ["item_id"],                       # 10
    ["item_id", "state_id"],           # 11
    ["item_id", "store_id"],           # 12
]

def cols_days(start: int, end: int) -> list[str]:
    return [f"d_{d}" for d in range(start, end + 1)]

def series_scale_for_group(wide: pd.DataFrame, group_cols: list[str]) -> pd.Series:
    """
    denominator for RMSSE on each series in a level:
      scale = mean( diff(y_train)^2 )
    where y_train is the aggregated series over d_1..d_1913
    """
    day_cols = cols_days(1, TR_LAST)
    if group_cols:
        gsum = wide.groupby(group_cols, observed=True)[day_cols].sum()
    else:
        s = wide[day_cols].sum(axis=0)
        gsum = s.to_frame().T
        gsum.index = pd.Index(["ALL"], name="all")

    arr = gsum.to_numpy(dtype=np.float64)
    if arr.shape[1] < 2:
        scale = np.ones(arr.shape[0], dtype=np.float64)
    else:
        diff = np.diff(arr, axis=1)
        scale = (diff ** 2).mean(axis=1)
        scale[scale == 0.0] = 1e-9
    return pd.Series(scale, index=gsum.index)

def weights_for_group(wide_eval: pd.DataFrame, cal: pd.DataFrame, prices: pd.DataFrame, group_cols: list[str]) -> pd.Series:
    """
    level weights: share of revenue in the last 28 days of training (d_1886..d_1913)
    revenue = sales * sell_price
    """
    last28 = cols_days(TR_LAST - H + 1, TR_LAST)  # d_1886..d_1913

    part = wide_eval[["id", "item_id", "store_id", "state_id", "dept_id", "cat_id"] + last28].copy()
    df = part.melt(
        id_vars=["id", "item_id", "store_id", "state_id", "dept_id", "cat_id"],
        value_vars=last28,
        var_name="d",
        value_name="sales",
    )
    df = df.merge(cal[["d", "wm_yr_wk"]], on="d", how="left")
    df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    df["sell_price"] = df["sell_price"].fillna(0.0)
    df["dollar"] = df["sales"].astype(np.float64) * df["sell_price"].astype(np.float64)

    if group_cols:
        gv = df.groupby(group_cols, observed=True)["dollar"].sum()
    else:
        gv = pd.Series(df["dollar"].sum(), index=pd.Index(["ALL"], name="all"))

    total = gv.sum()
    if total == 0.0:
        total = 1e-9
    return gv / total

def agg_matrix(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    """
    return a matrix with rows = groups, cols = d_1914..d_1941
    """
    if group_cols:
        key_cols = group_cols
    else:
        key_cols = ["__all__"]
        df = df.copy()
        df["__all__"] = "ALL"

    if df.empty:
        return pd.DataFrame()

    mat = df.groupby(key_cols + ["d"], observed=True)[value_col].sum().unstack("d").fillna(0.0)
    if "__all__" in mat.index.names:
        mat.index = pd.Index(["ALL"], name="all")
    # ensure ordered columns (d_1914..d_1941)
    all_eval_days = cols_days(TR_LAST + 1, TR_LAST + H)
    mat = mat.reindex(columns=all_eval_days, fill_value=0.0)
    return mat

def main():
    # 1) load submission (use evaluation rows only)
    sub = pd.read_csv(SUB_PATH)
    fcols = [c for c in sub.columns if c.startswith("F")]
    if len(fcols) != H:
        raise ValueError(f"submission must have F1..F{H} columns")

    sub_eval = sub[sub["id"].str.endswith("evaluation")].copy()

    pred_long = sub_eval.melt(id_vars=["id"], value_vars=fcols, var_name="F", value_name="sales_pred")
    pred_long["k"] = pred_long["F"].str[1:].astype(int)  # 1..28
    pred_long["d"] = "d_" + (TR_LAST + pred_long["k"]).astype(str)
    pred_long = pred_long.drop(columns=["F", "k"])

    # 2) load truth (evaluation window) with dimensions
    eval_wide = pd.read_csv(
        BASE_DIR / "sales_train_evaluation.csv",
        dtype={"item_id": "category", "dept_id": "category", "cat_id": "category", "store_id": "category", "state_id": "category"},
    )
    eval_days = cols_days(TR_LAST + 1, TR_LAST + H)
    truth_long = eval_wide.melt(
        id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
        value_vars=eval_days,
        var_name="d",
        value_name="sales_true",
    )

    # 3) merge pred with dimensions and truth
    pred_long = pred_long.merge(eval_wide[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]], on="id", how="left")
    df = pred_long.merge(truth_long[["id", "d", "sales_true"]], on=["id", "d"], how="left")

    # 4) load training-wide (for scale) + calendar/prices (for weights)
    wide_train = pd.read_csv(
        BASE_DIR / "sales_train_validation.csv",
        dtype={"item_id": "category", "dept_id": "category", "cat_id": "category", "store_id": "category", "state_id": "category"},
    )
    cal = pd.read_csv(BASE_DIR / "calendar.csv", usecols=["d", "wm_yr_wk"])
    prices = pd.read_csv(BASE_DIR / "sell_prices.csv", usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"])

    # 5) compute per-level scores
    results = {}
    level_scores = []
    for i, group_cols in enumerate(HIER_LIST, start=1):
        pred_mat = agg_matrix(df[["id", "d"] + group_cols + ["sales_pred"]], group_cols, "sales_pred")
        true_mat = agg_matrix(df[["id", "d"] + group_cols + ["sales_true"]], group_cols, "sales_true")

        if pred_mat.empty or true_mat.empty:
            results[f"level_{i}"] = float("nan")
            print(f"level_{i}: NaN (empty)")
            continue

        # align rows
        pred_mat, true_mat = pred_mat.align(true_mat, join="inner", axis=0)
        if pred_mat.empty:
            results[f"level_{i}"] = float("nan")
            print(f"level_{i}: NaN (no common groups)")
            continue

        err = pred_mat.to_numpy(np.float64) - true_mat.to_numpy(np.float64)
        mse_h = (err ** 2).mean(axis=1)  # per-series MSE over 28 days

        scale = series_scale_for_group(wide_train, group_cols).reindex(pred_mat.index)
        scale = scale.to_numpy(np.float64)
        scale[scale == 0.0] = 1e-9

        rmsse = np.sqrt(mse_h / scale)

        weights = weights_for_group(eval_wide, cal, prices, group_cols).reindex(pred_mat.index)
        w = weights.to_numpy(np.float64)
        w = np.where(np.isnan(w), 0.0, w)

        score = float((rmsse * w).sum())
        results[f"level_{i}"] = score
        level_scores.append(score)
        print(f"level_{i}: {score:.6f}")

    wrmsse = float(np.nanmean(level_scores)) if level_scores else float("nan")
    results["wrmsse"] = wrmsse
    print(f"\nWRMSSE (avg 12 levels): {wrmsse:.6f}")

if __name__ == "__main__":
    main()
