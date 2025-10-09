# /data/weizhen/code/math/train.py
import sys
import os
import gc
import json
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.append("/data/weizhen/code")
from data_preprocessing import create_dt, create_fea, H, MAX_LAGS, TR_LAST, FDAY


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Train LightGBM for M5 (accuracy track) with JSON config")
    parser.add_argument(
        "--config",
        type=str,
        default="/data/weizhen/code/math/config.json",
        help="Path to config.json"
    )
    # 可选：命令行覆盖config（不需要就别传）
    parser.add_argument("--model_out", type=str, default=None, help="Override model output path")
    parser.add_argument("--meta_out", type=str, default=None, help="Override meta output path")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    base_dir = cfg.get("base_dir")

    # 不从d_1开始, 用1-349作为趋势分析
    first_day = int(cfg.get("first_day", 350))
    nrows = cfg.get("nrows", None)
    nrows = None if nrows in (None, "None") else int(nrows)

    valid_size = int(cfg.get("valid_size"))
    seed = int(cfg.get("seed"))
    use_gpu = bool(cfg.get("use_gpu"))

    # 输出路径
    model_out = args.model_out or cfg.get("model_out", "model.lgb")
    meta_out = args.meta_out or cfg.get("meta_out", "model_meta.json")

    # LightGBM 参数
    lgbm_cfg = cfg.get("lgbm_params", {})
    learning_rate = float(lgbm_cfg.get("learning_rate", 0.075))
    num_iterations = int(lgbm_cfg.get("num_iterations", 1200))
    num_leaves = int(lgbm_cfg.get("num_leaves", 128))
    min_data_in_leaf = int(lgbm_cfg.get("min_data_in_leaf", 100))
    bagging_fraction = float(lgbm_cfg.get("bagging_fraction", 0.75))
    lambda_l2 = float(lgbm_cfg.get("lambda_l2", 0.1))

    params = {
        "objective": lgbm_cfg.get("objective", "poisson"),
        "metric": lgbm_cfg.get("metric", ["rmse"]),
        "force_row_wise": lgbm_cfg.get("force_row_wise", True),
        "learning_rate": learning_rate,
        "sub_row": lgbm_cfg.get("sub_row", bagging_fraction),  # 兼容旧字段
        "bagging_fraction": bagging_fraction,
        "bagging_freq": int(lgbm_cfg.get("bagging_freq", 1)),
        "lambda_l2": lambda_l2,
        "verbosity": int(lgbm_cfg.get("verbosity", 1)),
        "num_iterations": num_iterations,
        "num_leaves": num_leaves,
        "min_data_in_leaf": min_data_in_leaf,
    }
    if use_gpu or lgbm_cfg.get("use_gpu"):
        params.update({"device_type": "gpu"})

    # 固定随机种子
    np.random.seed(seed)

    print("=== Config ===")
    print(f"config_path={args.config}")
    print(f"base_dir={base_dir}")
    print(f"first_day={first_day}, nrows={nrows}, valid_size={valid_size}")
    print(f"seed={seed}, use_gpu={use_gpu}")
    print(f"model_out={model_out}, meta_out={meta_out}")
    print("LightGBM params:", params)
    print()

    # 1) 读取并整理 -> 长表
    print("Loading & merging data ...")
    df = create_dt(is_train=True, first_day=first_day, nrows=nrows, base_dir=base_dir)
    print("Raw long df:", df.shape)

    # 2) 构造特征
    print("Creating features (lags, rolling means, date features) ...")
    create_fea(df)
    print("With features:", df.shape)

    # 3) 丢掉因滞后/滚动产生的 NaN，避免前28天没数据
    before = df.shape[0]
    df.dropna(inplace=True)
    after = df.shape[0]
    print(f"Dropna: {before} -> {after}")

    # [
    #     'id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
    #     'sales', 'd', 'wm_yr_wk', 'weekday', 'wday', 'month', 'year',
    #     'event_name_1', 'event_name_2', 'event_type_1', 'event_type_2',
    #     'snap_CA', 'snap_TX', 'snap_WI', 'sell_price',
    #     'lag_7', 'lag_28', 'rmean_7_7', 'rmean_7_28',
    #     'rmean_28_7', 'rmean_28_28', 'week', 'quarter', 'mday'
    #     ]

    # 4) 特征列与标签指定哪些是x，哪些是y
    # 去除无用列后，剩下的都是x
    # y 为 sales
    cat_feats = [
        "item_id", "dept_id", "store_id", "cat_id", "state_id",
        "event_name_1", "event_name_2", "event_type_1", "event_type_2"
    ]
    useless_cols = ["id", "date", "sales", "d", "wm_yr_wk", "weekday"]
    train_cols = df.columns[~df.columns.isin(useless_cols)]
    X_train = df[train_cols]
    y_train = df["sales"]
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("Num features:", len(train_cols))

    # 5) 构造“伪验证集”
    total = X_train.index.values
    if valid_size >= len(total) - 1:
        valid_size = max(1, len(total) // 10)
        print(f"[warn] valid_size too large, fallback to {valid_size}")
    fake_valid_inds = np.random.choice(total, valid_size, replace=False)
    train_inds = np.setdiff1d(total, fake_valid_inds)

    train_data = lgb.Dataset(
        X_train.loc[train_inds], label=y_train.loc[train_inds],
        categorical_feature=cat_feats, free_raw_data=False
    )
    fake_valid_data = lgb.Dataset(
        X_train.loc[fake_valid_inds], label=y_train.loc[fake_valid_inds],
        categorical_feature=cat_feats, free_raw_data=False
    )

    # 6) 训练
    print("Start training ...")
    num_boost_round = int(params.pop("num_iterations", 1200))
    callbacks = [lgb.log_evaluation(50)]

    m_lgb = lgb.train(
        params=params,
        train_set=train_data,
        valid_sets=[fake_valid_data],
        valid_names=["valid"],
        num_boost_round=num_boost_round,
        callbacks=callbacks
    )

    # 7) 保存模型与元数据
    print(f"Saving model to: {model_out}")
    m_lgb.save_model(model_out)

    meta = {
        "train_cols": list(train_cols),
        "cat_feats": list(cat_feats),
        "first_day": first_day,
        "base_dir": base_dir,
        "H": H,
        "MAX_LAGS": MAX_LAGS,
        "TR_LAST": TR_LAST,
        "FDAY": FDAY.strftime("%Y-%m-%d"),
        "params": params,
        "seed": seed,
    }
    print(f"Saving meta to: {meta_out}")
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 8) 清理
    del df, X_train, y_train, train_data, fake_valid_data
    gc.collect()
    print("Done.")


if __name__ == "__main__":
    main()
