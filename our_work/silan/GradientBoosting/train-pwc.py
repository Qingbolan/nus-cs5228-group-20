import pickle
import time  # 用于记录时间
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

# 设定模型参数
model_GBoost = GradientBoostingRegressor(
    n_estimators=3000,
    learning_rate=0.1,
    max_depth=5,
    max_features='sqrt',
    min_samples_leaf=20,
    min_samples_split=15,
    loss='huber',
    random_state=42
)


# 定义RMSE计算函数
def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# K-fold Cross Validation
def rmsle_cross_val(X_train, y_train, n_folds, model):
    kf = KFold(n_splits=n_folds, shuffle=True,random_state=42)  # 将数据集分成n_splits份，shuffle=True表示打乱数据集，random_state=42表示随机种子
    rmse_scores = []

    # 输出每个fold的RMSE
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_fold_train, X_fold_val = X_train.values[train_idx], X_train.values[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

        # 开始训练和评估
        model.fit(X_fold_train, y_fold_train)
        y_fold_pred = model.predict(X_fold_val)

        # 计算当前fold的RMSE
        fold_rmse = rmse(y_fold_val, y_fold_pred)
        rmse_scores.append(fold_rmse)

        print(f"Fold {fold_idx + 1}: RMSE = {fold_rmse:.4f}")

    return np.array(rmse_scores)


if __name__ == '__main__':
    # 训练集数据传入
    train_data = pd.read_csv('preprocessing/2024-10-10-silan/train_cleaned.csv')
    # 删除第0列
    train_data = train_data.drop(columns=train_data.columns[0])

    train_labels = train_data['price']  # 假设标签列为 'price'
    train_data = train_data.drop(columns=['price'])  # 删除目标列

    # 创建 LabelEncoder 对象
    le_make = LabelEncoder()
    le_model = LabelEncoder()

    # 对 'make' 列进行标签编码
    train_data['make'] = le_make.fit_transform(train_data['make'])

    # 对 'model' 列进行标签编码
    train_data['model'] = le_model.fit_transform(train_data['model'])

    # 统计数据的行和列
    print(f"Training data has {train_data.shape[0]} rows and {train_data.shape[1]} columns.")

    # 开始计时
    start_time = time.time()

    # 使用K折交叉验证评估模型性能
    score_GBoost = rmsle_cross_val(train_data, np.log(train_labels), 5, model_GBoost)
    print(f"GradientBoostingRegressor average score: {score_GBoost.mean():.4f} ({score_GBoost.std():.4f})\n")

    # 结束计时，计算训练时间
    elapsed_time = time.time() - start_time
    print(f"Time taken for training: {elapsed_time:.2f} seconds")

    # 最终在整个数据集上训练模型
    model_GBoost.fit(train_data.values, train_labels.values)

    # 保存模型
    model_path = 'gboost_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_GBoost, f)

    print(f"Training Completed. Model Saved at {model_path}")