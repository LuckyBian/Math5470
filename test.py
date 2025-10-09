import pandas as pd

base_dir = "/data/weizhen/code/math"

# 读取预测结果（即 infer 阶段生成的 submission.csv 或 te DataFrame）
pred = pd.read_csv(f"{base_dir}/submission.csv")
print("=== Predicted Columns ===")
print(pred.columns.tolist())
print(pred.head(3))

# 读取真实值文件
truth = pd.read_csv(f"{base_dir}/sales_train_evaluation.csv", nrows=1)
print("\n=== Truth Columns ===")
print(truth.columns.tolist())

# 检查必要分组列
required_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
missing_pred = [c for c in required_cols if c not in pred.columns]
missing_truth = [c for c in required_cols if c not in truth.columns]

print("\n=== Column Check ===")
if not missing_pred and not missing_truth:
    print("✅ Both prediction and truth contain all required group columns.")
else:
    if missing_pred:
        print(f"❌ Missing in prediction: {missing_pred}")
    if missing_truth:
        print(f"❌ Missing in truth: {missing_truth}")

# 检查 id 对齐情况
pred_ids = set(pred["id"].str.replace("validation$", "evaluation", regex=True))
truth_ids = set(truth["id"])
missing_in_truth = pred_ids - truth_ids
missing_in_pred = truth_ids - pred_ids

print("\n=== ID Alignment Check ===")
print(f"Prediction IDs not in truth: {len(missing_in_truth)}")
print(f"Truth IDs not in prediction: {len(missing_in_pred)}")
if len(missing_in_truth) < 10:
    print(list(missing_in_truth)[:10])
