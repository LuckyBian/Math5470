# /home/liyiming/math/train_xgboost.py
import sys
import os
import gc
import json
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.append("/home/liyiming")
from data_preprocessing import create_dt, create_fea, H, MAX_LAGS, TR_LAST, FDAY


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Train XGBoost for M5 (accuracy track) with JSON config")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/liyiming/math/config.json",
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

    # 输出路径 - 修改默认扩展名为 .xgb
    model_out = args.model_out or cfg.get("model_out", "model.lgb").replace(".lgb", ".xgb")
    meta_out = args.meta_out or cfg.get("meta_out", "model_meta_xgboost.json")

    # XGBoost 参数 - 从 lgbm_params 转换为 xgboost_params
    xgb_cfg = cfg.get("xgboost_params", cfg.get("lgbm_params", {}))
    learning_rate = float(xgb_cfg.get("learning_rate", 0.075))
    n_estimators = int(xgb_cfg.get("n_estimators", xgb_cfg.get("num_iterations", 1200)))
    max_depth = int(xgb_cfg.get("max_depth", 6))  # XGBoost 使用 max_depth 而不是 num_leaves
    min_child_weight = int(xgb_cfg.get("min_child_weight", xgb_cfg.get("min_data_in_leaf", 100)))
    subsample = float(xgb_cfg.get("subsample", xgb_cfg.get("bagging_fraction", 0.75)))
    reg_lambda = float(xgb_cfg.get("reg_lambda", xgb_cfg.get("lambda_l2", 0.1)))

    params = {
        "objective": "count:poisson" if xgb_cfg.get("objective") == "poisson" else "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": learning_rate,
        "subsample": subsample,
        "reg_lambda": reg_lambda,
        "random_state": seed,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "verbosity": int(xgb_cfg.get("verbosity", 1)),
        "n_jobs": -1,  # 使用所有CPU核心
        "early_stopping_rounds": 100,  # 早停轮数
    }
    
    # GPU 支持
    if use_gpu or xgb_cfg.get("use_gpu"):
        params.update({"tree_method": "gpu_hist", "gpu_id": 0})
    else:
        params.update({"tree_method": "hist"})

    # 固定随机种子
    np.random.seed(seed)

    print("=== Config ===")
    print(f"config_path={args.config}")
    print(f"base_dir={base_dir}")
    print(f"first_day={first_day}, nrows={nrows}, valid_size={valid_size}")
    print(f"seed={seed}, use_gpu={use_gpu}")
    print(f"model_out={model_out}, meta_out={meta_out}")
    print("XGBoost params:", params)
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
    print(X_train.columns)
    print(X_train.head(1))
    pd.DataFrame(X_train[:10000]).to_csv("X_train.csv", index=False)
    y_train = df["sales"]
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("Num features:", len(train_cols))

    # 处理分类特征 - XGBoost 需要数值编码
    print("Encoding categorical features ...")
    from sklearn.preprocessing import LabelEncoder
    label_encoders = {}
    X_train_encoded = X_train.copy()
    
    for col in cat_feats:
        if col in X_train_encoded.columns:
            le = LabelEncoder()
            X_train_encoded[col] = le.fit_transform(X_train_encoded[col].astype(str))
            label_encoders[col] = le
    
    print("Categorical features encoded.")

    # 5) 构造"伪验证集"
    total = X_train_encoded.index.values
    if valid_size >= len(total) - 1:
        valid_size = max(1, len(total) // 10)
        print(f"[warn] valid_size too large, fallback to {valid_size}")
    fake_valid_inds = np.random.choice(total, valid_size, replace=False)
    train_inds = np.setdiff1d(total, fake_valid_inds)

    X_train_split = X_train_encoded.loc[train_inds]
    y_train_split = y_train.loc[train_inds]
    X_valid = X_train_encoded.loc[fake_valid_inds]
    y_valid = y_train.loc[fake_valid_inds]

    # 6) 训练
    print("Start training ...")
    
    # 创建 XGBoost 模型
    xgb_model = xgb.XGBRegressor(**params)
    
    # 训练模型，使用验证集
    xgb_model.fit(
        X_train_split, y_train_split,
        eval_set=[(X_valid, y_valid)],
        verbose=50  # 每50轮输出一次
    )

    # 7) 保存模型与元数据
    print(f"Saving model to: {model_out}")
    xgb_model.save_model(model_out)

    meta = {
        "train_cols": list(train_cols),
        "cat_feats": list(cat_feats),
        "label_encoders": {k: list(v.classes_) for k, v in label_encoders.items()},
        "first_day": first_day,
        "base_dir": base_dir,
        "H": H,
        "MAX_LAGS": MAX_LAGS,
        "TR_LAST": TR_LAST,
        "FDAY": FDAY.strftime("%Y-%m-%d"),
        "params": params,
        "seed": seed,
        "model_type": "xgboost"
    }
    print(f"Saving meta to: {meta_out}")
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 8) 清理
    del df, X_train, X_train_encoded, y_train, X_train_split, y_train_split, X_valid, y_valid
    gc.collect()
    print("Done.")


if __name__ == "__main__":
    main()
