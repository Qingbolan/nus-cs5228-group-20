import pandas as pd
import numpy as np
from mlbox.preprocessing import Reader, Drift_thresholder, NA_encoder
from mlbox.optimisation import Optimiser
from mlbox.prediction import Predictor
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# 设置并行计算的核心数
N_JOBS = os.cpu_count()  # 获取CPU核心数
N_JOBS = max(1, N_JOBS - 1)  # 保留一个核心给系统

def calculate_num_leaves(max_depth):
    """根据max_depth计算合适的num_leaves"""
    return min(2**max_depth - 1, 1000)  # 设置一个上限以避免内存问题

# 定义文件路径和目标变量
train_path = 'train_cleaned.csv'
test_path = 'test_cleaned.csv'
target_name = 'price'
paths = [train_path, test_path]

# 数据读取和预处理
reader = Reader(sep=',')
data = reader.train_test_split(paths, target_name)

# 应用漂移阈值处理
drift_thresholder = Drift_thresholder()
data = drift_thresholder.fit_transform(data)

# 处理缺失值
na_encoder = NA_encoder()
data['train'] = na_encoder.fit_transform(data['train'], data['target'])
data['test'] = na_encoder.transform(data['test'])

# 优化后的参数空间
space = {
    # 数值特征处理策略
    'ne__numerical_strategy': {'search': 'choice', 'space': ['mean', 'median']},
    
    # 分类特征编码策略
    'ce__strategy': {'search': 'choice', 'space': ['label_encoding']},
    
    # 特征选择
    'fs__strategy': {'search': 'choice', 'space': ['rf_feature_importance']},
    'fs__threshold': {'search': 'uniform', 'space': [0.01, 0.02]},
    
    # LightGBM 参数
    'est__strategy': {'search': 'choice', 'space': ['LightGBM']},
    'est__boosting_type': {'search': 'choice', 'space': ['gbdt']},
    
    # 调整深度和叶子数的关系
    'est__max_depth': {'search': 'choice', 'space': [3, 4, 5]},
    # 确保num_leaves小于2^max_depth
    'est__num_leaves': {'search': 'choice', 'space': [calculate_num_leaves(depth) for depth in [3, 4, 5]]},
    
    # 统一线程设置
    'est__num_threads': {'search': 'choice', 'space': [N_JOBS]},
    
    # 关键参数
    'est__min_child_samples': {'search': 'choice', 'space': [50, 100]},
    'est__learning_rate': {'search': 'uniform', 'space': [0.01, 0.02]},
    'est__n_estimators': {'search': 'choice', 'space': [2000]},
    
    # 采样参数
    'est__subsample': {'search': 'uniform', 'space': [0.7, 0.8]},
    'est__colsample_bytree': {'search': 'uniform', 'space': [0.7, 0.8]},
    'est__subsample_freq': {'search': 'choice', 'space': [5]},
    
    # 正则化参数
    'est__reg_alpha': {'search': 'uniform', 'space': [3.0, 4.0]},
    'est__reg_lambda': {'search': 'uniform', 'space': [6.0, 7.0]},
    
    # 固定参数
    'est__objective': {'search': 'choice', 'space': ['regression']},
    'est__metric': {'search': 'choice', 'space': ['rmse']},
    'est__min_child_weight': {'search': 'choice', 'space': [0.001]},
    'est__min_split_gain': {'search': 'choice', 'space': [0.0]},
    'est__verbose': {'search': 'choice', 'space': [-1]},  # 减少输出
}

# 创建优化器
opt = Optimiser(
    scoring='neg_root_mean_squared_error',
    n_folds=5
)

print("\nStarting optimization with {} trials...".format(20))  # 减少试验次数以加快过程
print(f"Using {N_JOBS} CPU cores for LightGBM...")

# 进行优化
best_params = opt.optimise(space, data, max_evals=20)

# 输出最佳参数的详细信息
print("\nBest parameters found:")
for param, value in best_params.items():
    print(f"{param}: {value}")

# 创建预测器
predictor = Predictor()

# 进行拟合和预测，并获取预测结果
predictions = predictor.fit_predict(best_params, data)

# 检查 predictions 的类型和内容
print(f"Type of predictions: {type(predictions)}")
print(predictions)

# 正确处理预测结果
if isinstance(predictions, pd.DataFrame):
    submission = pd.DataFrame({
        'Id': range(len(predictions)),
        'Predicted': predictions[predictions.columns[0]].clip(700, 2900000).round().astype(int)
    })
else:
    # 如果 predictions 是 numpy 数组
    submission = pd.DataFrame({
        'Id': range(len(predictions)),
        'Predicted': np.clip(predictions, 700, 2900000).round().astype(int)
    })

# Save or process `submission` as needed.


# 输出预测统计信息
print("\nPrediction Summary:")
print("\nBasic Statistics:")
print(submission['Predicted'].describe())

print("\nValue Range Check:")
print(f"Minimum value: {submission['Predicted'].min()}")
print(f"Maximum value: {submission['Predicted'].max()}")
print(f"Number of predictions: {len(submission)}")

print("\nDistribution Analysis:")
print(f"Number of minimum value (700) predictions: {(submission['Predicted'] == 700).sum()}")
print(f"Number of maximum value (2900000) predictions: {(submission['Predicted'] == 2900000).sum()}")

# 计算四分位数
q1 = submission['Predicted'].quantile(0.25)
q3 = submission['Predicted'].quantile(0.75)
iqr = q3 - q1
print(f"\nQuartile Analysis:")
print(f"Q1 (25th percentile): {q1:,.2f}")
print(f"Q3 (75th percentile): {q3:,.2f}")
print(f"IQR: {iqr:,.2f}")

# 保存预测结果
submission.to_csv('submission_mlbox.csv', index=False)
print("\nPredictions saved to 'submission_mlbox.csv'")

# 保存详细信息
with open('model_details.txt', 'w') as f:
    f.write("Best Parameters:\n")
    for param, value in best_params.items():
        f.write(f"{param}: {value}\n")
    
    f.write("\nPrediction Statistics:\n")
    f.write(str(submission['Predicted'].describe()))
    
    f.write("\n\nValue Distribution:\n")
    f.write(f"Minimum value: {submission['Predicted'].min()}\n")
    f.write(f"Maximum value: {submission['Predicted'].max()}\n")
    f.write(f"Number of minimum value predictions: {(submission['Predicted'] == 700).sum()}\n")
    f.write(f"Number of maximum value predictions: {(submission['Predicted'] == 2900000).sum()}\n")
    
    f.write("\nQuartile Analysis:\n")
    f.write(f"Q1: {q1:,.2f}\n")
    f.write(f"Q3: {q3:,.2f}\n")
    f.write(f"IQR: {iqr:,.2f}\n")

print("\nDetailed model information saved to 'model_details.txt'")

# 可选：创建预测值分布图
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    
    # 主直方图
    plt.hist(submission['Predicted'], bins=50, edgecolor='black')
    plt.title('Distribution of Predicted Values')
    plt.xlabel('Predicted Price')
    plt.ylabel('Frequency')
    
    # 添加四分位线
    plt.axvline(x=q1, color='r', linestyle='--', alpha=0.5, label='Q1')
    plt.axvline(x=q3, color='g', linestyle='--', alpha=0.5, label='Q3')
    plt.axvline(x=submission['Predicted'].median(), color='b', linestyle='--', alpha=0.5, label='Median')
    
    plt.legend()
    plt.savefig('prediction_distribution.png')
    plt.close()
    print("\nPrediction distribution plot saved as 'prediction_distribution.png'")
except ImportError:
    print("\nMatplotlib not available - skipping distribution plot")
except Exception as e:
    print(f"\nError creating distribution plot: {str(e)}")
