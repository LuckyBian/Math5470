# /data/weizhen/code/math/infer.py
from __future__ import annotations

import json
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import lightgbm as lgb

from data_preprocessing import create_dt, create_fea, H, MAX_LAGS, FDAY  # TR_LAST 不再需要


# ------------------ 基础加载 ------------------
def load_meta(meta_path: str):
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(model_path: str) -> lgb.Booster:
    return lgb.Booster(model_file=model_path)


# ------------------ 生成 submission ------------------
def make_submission_like(te: pd.DataFrame, fday, cols, out_csv: str) -> pd.DataFrame:
    """
    构建 M5 预测提交文件 (submission.csv)
    包含:
      - 仅保留未来28天 (fday之后)
      - 保留分组列(item_id, dept_id, cat_id, store_id, state_id)
      - 构造 F1-F28 列
      - validation / evaluation 双版本拼接
    """
    # 保留所有层级列以支持后续评估或检查
    te_sub = te.loc[
        te.date >= fday,
        ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "sales"]
    ].copy()

    # 构造 F1–F28 列（每个id累积计数）
    te_sub["F"] = te_sub.groupby("id").cumcount().add(1).map(lambda k: f"F{k}")

    # 生成 Kaggle 官方格式：每个 id 对应 F1–F28 一行
    te_sub = (
        te_sub.set_index(["id", "F"])
        .unstack()["sales"][cols]
        .reset_index()
        .fillna(0.0)
        .sort_values("id")
        .reset_index(drop=True)
    )

    # 构造 evaluation 集（替换 id 后缀）
    sub2 = te_sub.copy()
    sub2["id"] = sub2["id"].str.replace("validation$", "evaluation", regex=True)

    # 拼接 validation + evaluation
    sub = pd.concat([te_sub, sub2], axis=0, sort=False)

    # 保存为 CSV
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_csv, index=False)
    print(f"✅ Saved submission to {out_csv}")

    return sub


# ------------------ 主流程：仅推理并输出提交 ------------------
def main():
    model_path = "/data/weizhen/code/math/model.lgb"
    meta_path = "/data/weizhen/code/math/model_meta.json"
    out_path = "/data/weizhen/code/math/submission.csv"

    meta = load_meta(meta_path)
    base_dir = meta.get("base_dir", "/data/weizhen/code/math")
    train_cols = meta["train_cols"]
    model = load_model(model_path)

    print("=== Inference (rolling 28d) ===")
    print(f"base_dir={base_dir}")
    print(f"FDAY={FDAY.date()}, H={H}, MAX_LAGS={MAX_LAGS}")
    print(f"#features={len(train_cols)}: {train_cols[:8]} ...")
    print()

    # 1) 推理底表（含历史窗口与未来空位）
    te = create_dt(is_train=False, first_day=meta.get("first_day", 350), base_dir=base_dir)

    # 2) 逐日滚动预测
    for t in range(H):
        day = FDAY + timedelta(days=t)
        wnd = te[(te.date >= day - timedelta(days=MAX_LAGS)) & (te.date <= day)].copy()
        create_fea(wnd)
        X_t = wnd.loc[wnd.date == day, train_cols]
        te.loc[te.date == day, "sales"] = model.predict(X_t)
        if (t + 1) % 7 == 0:
            print(f"  done {t + 1}/{H} days → {day.date()}")

    # 3) 生成 submission.csv
    cols = [f"F{i}" for i in range(1, H + 1)]
    make_submission_like(te, FDAY, cols, out_csv=out_path)


if __name__ == "__main__":
    main()
