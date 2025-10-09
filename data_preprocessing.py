# /data/weizhen/code/math/data_preprocessing.py
from __future__ import annotations

from datetime import datetime
from typing import Optional, Iterable

import numpy as np
import pandas as pd

# ========= 全局常量（与原笔记保持一致） =========
H: int = 28                 # 预测步长：未来 28 天
MAX_LAGS: int = 57          # 最大滞后跨度（滚动推理用）
TR_LAST: int = 1913         # 训练集最后一天的 d 索引（d_1..d_1913）
FDAY: datetime = datetime(2016, 4, 25)  # d_1914 对应的实际日期

# ========= CSV 列类型定义（节省内存） =========
CAL_DTYPES = {
    "event_name_1": "category",
    "event_name_2": "category",
    "event_type_1": "category",
    "event_type_2": "category",
    "weekday": "category",
    "wm_yr_wk": "int16",
    "wday": "int16",
    "month": "int16",
    "year": "int16",
    "snap_CA": "float32",
    "snap_TX": "float32",
    "snap_WI": "float32",
}

PRICE_DTYPES = {
    "store_id": "category",
    "item_id": "category",
    "wm_yr_wk": "int16",
    "sell_price": "float32",
}


def _encode_category_inplace(df: pd.DataFrame, cols: Iterable[str]) -> None:
    """
    将指定列转换为唯一的整数，节省内存，方便处理
    """
    for col in cols:
        if col not in df.columns:
            continue
        if str(df[col].dtype) != "category":
            df[col] = df[col].astype("category")
        codes = df[col].cat.codes.astype("int16")
        # 保证最小值从0开始（与原笔记一致）
        if len(codes) > 0:
            min_code = codes.min()
            if min_code != 0 and min_code != -1:  # -1 表示 NaN
                codes = (codes - min_code).astype("int16")
        df[col] = codes


def _load_calendar(base_dir: str) -> pd.DataFrame:
    """
    读取 calendar.csv，转换日期，并对标注为 category 的列进行紧凑编码。
    ["event_name_1", "event_name_2", "event_type_1", "event_type_2", "weekday"]
    把星期和节日及类型进行编码
    """
    cal = pd.read_csv(f"{base_dir}/calendar.csv", dtype=CAL_DTYPES)
    cal["date"] = pd.to_datetime(cal["date"])
    cat_cols = [c for c, t in CAL_DTYPES.items() if t == "category"]
    _encode_category_inplace(cal, cat_cols)
    return cal


def _load_prices(base_dir: str) -> pd.DataFrame:
    """
    读取 sell_prices.csv，并对 store_id / item_id 做紧凑编码。
    """
    prices = pd.read_csv(f"{base_dir}/sell_prices.csv", dtype=PRICE_DTYPES)
    _encode_category_inplace(prices, ["store_id", "item_id"])
    return prices


def create_dt(
    is_train: bool = True,
    nrows: Optional[int] = None,
    first_day: int = 1200,
    base_dir: str = "/data/weizhen/code/math",
) -> pd.DataFrame:
    """
    读取销量宽表 -> 选取列(d_first_day..d_TR_LAST) -> melt成长表，
    并与 calendar / sell_prices 合并，返回“特征+标签”一体的长表 DataFrame。

    参数:
        is_train : True=训练/验证；False=推理（会额外补充 d_1914..d_1941 空列）
        nrows    : 读取销量宽表的行数（调试/抽样）；None=全部
        first_day: 从哪天开始抽取 d_* 列（越小行数越多，内存越大；常用 350/1000/1200）
        base_dir : 数据目录（包含 calendar.csv / sell_prices.csv / sales_train_validation.csv）
    """
    prices = _load_prices(base_dir)
    cal = _load_calendar(base_dir)

    # 选取销量列范围
    start_day = max(1 if is_train else TR_LAST - MAX_LAGS, first_day) # 350
    numcols = [f"d_{day}" for day in range(start_day, TR_LAST + 1)] # d_350..d_1913
    catcols = ["id", "item_id", "dept_id", "store_id", "cat_id", "state_id"]

    # 读取整张表，从d_350开始读
    dtype = {c: "float32" for c in numcols}
    dtype.update({c: "category" for c in catcols if c != "id"})
    sales = pd.read_csv(
        f"{base_dir}/sales_train_validation.csv",
        nrows=nrows,
        usecols=catcols + numcols,
        dtype=dtype,
    )

    # 所有地点，店铺信息改为id
    _encode_category_inplace(sales, [c for c in catcols if c != "id"])

    # 推理阶段，设置后面的一些天的值为none
    if not is_train:
        for day in range(TR_LAST + 1, TR_LAST + H + 1):
            sales[f"d_{day}"] = np.nan

    # 宽转长
    dt = pd.melt(
        sales,
        id_vars=catcols,
        value_vars=[c for c in sales.columns if c.startswith("d_")],
        var_name="d",
        value_name="sales",
    )

    # 合并 calendar（按 d
    dt = dt.merge(cal, on="d", copy=False)
    dt = dt.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], copy=False)

    # 返回一个大表，某个商品在某一天的销量，且这一天有什么节日，星期几，价格多少

    return dt


def _safe_week_number(series_dt: pd.Series) -> pd.Series:
    """
    兼容不同 pandas 版本：
    - 优先使用 .dt.isocalendar().week
    - 若不可用则回退到 .dt.week / .dt.weekofyear
    """
    try:
        # pandas >= 1.1：isocalendar() 返回 DataFrame[year, week, day]
        return series_dt.dt.isocalendar().week.astype("int16")
    except Exception:
        # 旧版本兼容
        if hasattr(series_dt.dt, "weekofyear"):
            return series_dt.dt.weekofyear.astype("int16")
        return series_dt.dt.week.astype("int16")


def create_fea(dt: pd.DataFrame) -> None:
    """
    就地 (in-place) 给 dt 添加基础特征：
      - 滞后特征: lag_7, lag_28
      - 基于滞后列的滚动均值: rmean_7_7, rmean_28_7, rmean_7_28, rmean_28_28（防泄露）
      - 日期特征: wday/week/month/quarter/year/mday（若原列缺失则由 'date' 计算）
    """
    #设置两个滞后特征，7天和28天
    lags = [7, 28]
    lag_cols = [f"lag_{lag}" for lag in lags]
    grp = dt.groupby("id", sort=False)["sales"]

    # 多出两列，lag_7和lag_28,表示7天前和28天前商品的销量
    for lag, lag_col in zip(lags, lag_cols):
        dt[lag_col] = grp.shift(lag).astype("float32")

    # ---- 2) 滚动均值----
    # 计算7天前7/28天的滚动均值
    wins = [7, 28]
    for win in wins:
        for lag, lag_col in zip(lags, lag_cols):
            dt[f"rmean_{lag}_{win}"] = (
                dt.groupby("id", sort=False)[lag_col]
                .transform(lambda x: x.rolling(win).mean())
                .astype("float32")
            )

    # ---- 3) 日期特征 ----
    # 计算日周月季年，捕捉这些特征。比如周末打折，商品卖的多
    if "wday" in dt.columns:
        dt["wday"] = dt["wday"].astype("int16")
    else:
        dt["wday"] = dt["date"].dt.weekday.astype("int16")

    # week
    if "week" in dt.columns:
        dt["week"] = dt["week"].astype("int16")
    else:
        dt["week"] = _safe_week_number(dt["date"])

    # month
    if "month" in dt.columns:
        dt["month"] = dt["month"].astype("int16")
    else:
        dt["month"] = dt["date"].dt.month.astype("int16")

    # quarter
    if "quarter" in dt.columns:
        dt["quarter"] = dt["quarter"].astype("int16")
    else:
        dt["quarter"] = dt["date"].dt.quarter.astype("int16")

    # year
    if "year" in dt.columns:
        dt["year"] = dt["year"].astype("int16")
    else:
        dt["year"] = dt["date"].dt.year.astype("int16")

    # mday（每月第几天）
    if "mday" in dt.columns:
        dt["mday"] = dt["mday"].astype("int16")
    else:
        dt["mday"] = dt["date"].dt.day.astype("int16")


__all__ = [
    "H",
    "MAX_LAGS",
    "TR_LAST",
    "FDAY",
    "create_dt",
    "create_fea",
]
