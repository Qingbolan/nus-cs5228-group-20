import pickle
import time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

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

def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def train_and_evaluate(train_idx, val_idx, X, y, model):
    X_fold_train, X_fold_val = X[train_idx], X[val_idx]
    y_fold_train, y_fold_val = y[train_idx], y[val_idx]
    
    model.fit(X_fold_train, y_fold_train)
    y_fold_pred = model.predict(X_fold_val)
    
    fold_rmse = rmse(y_fold_val, y_fold_pred)
    return fold_rmse

def rmsle_cross_val(X_train, y_train, n_folds, model, n_jobs=-1):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    rmse_scores = Parallel(n_jobs=n_jobs)(
        delayed(train_and_evaluate)(train_idx, val_idx, X_train, y_train, model)
        for train_idx, val_idx in kf.split(X_train)
    )
    
    for fold_idx, fold_rmse in enumerate(rmse_scores):
        print(f"Fold {fold_idx + 1}: RMSE = {fold_rmse:.4f}")
    
    return np.array(rmse_scores)

if __name__ == '__main__':
    # 读取数据
    train_data = pd.read_csv('preprocessing/release/ver2/train_cleaned.csv')

    train_data = train_data.drop(columns=train_data.columns[0])

    train_labels = train_data['price']
    train_data = train_data.drop(columns=['price'])

    # 特征工程
    # le = LabelEncoder()
    # for col in ['make', 'model']:
    #     train_data[col] = le.fit_transform(train_data[col])

    scaler = StandardScaler()
    numeric_features = ['curb_weight', 'power', 'engine_cap']
    train_data[numeric_features] = scaler.fit_transform(train_data[numeric_features])

    print(f"Training data has {train_data.shape[0]} rows and {train_data.shape[1]} columns.")

    start_time = time.time()

    # 使用K折交叉验证评估模型性能
    score_GBoost = rmsle_cross_val(train_data.values, np.log(train_labels), 5, model_GBoost)
    print(f"GradientBoostingRegressor average score: {score_GBoost.mean():.4f} ({score_GBoost.std():.4f})\n")

    elapsed_time = time.time() - start_time
    print(f"Time taken for training: {elapsed_time:.2f} seconds")

    # 最终在整个数据集上训练模型
    model_GBoost.fit(train_data.values, train_labels.values)

    # 保存模型
    model_path = 'gboost_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_GBoost, f)

    print(f"Training Completed. Model Saved at {model_path}")