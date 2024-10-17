import pickle
import time
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# 设定模型参数
model_CatBoost = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.1,
    depth=5,
    l2_leaf_reg=3,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    verbose=False
)

def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def train_and_evaluate(train_idx, val_idx, X, y, cat_features, model):
    X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
    y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
   
    train_pool = Pool(X_fold_train, y_fold_train, cat_features=cat_features)
    val_pool = Pool(X_fold_val, y_fold_val, cat_features=cat_features)
   
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100, verbose=False)
    y_fold_pred = model.predict(X_fold_val)
   
    fold_rmse = rmse(y_fold_val, y_fold_pred)
    return fold_rmse

def rmsle_cross_val(X_train, y_train, cat_features, n_folds, model, n_jobs=-1):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
   
    rmse_scores = Parallel(n_jobs=n_jobs)(
        delayed(train_and_evaluate)(train_idx, val_idx, X_train, y_train, cat_features, model)
        for train_idx, val_idx in kf.split(X_train)
    )
   
    for fold_idx, fold_rmse in enumerate(rmse_scores):
        print(f"Fold {fold_idx + 1}: RMSE = {fold_rmse:.4f}")
   
    return np.array(rmse_scores)

if __name__ == '__main__':
    # 读取数据
    train_data = pd.read_csv('preprocessing/2024-10-10-silan/train_cleaned.csv')
    train_data = train_data.drop(columns=train_data.columns[0])
    train_labels = train_data['price']
    train_data = train_data.drop(columns=['price'])

    # 特征工程
    cat_features = []
    numeric_features = ['curb_weight', 'power', 'engine_cap']

    scaler = StandardScaler()
    train_data[numeric_features] = scaler.fit_transform(train_data[numeric_features])

    print(f"Training data has {train_data.shape[0]} rows and {train_data.shape[1]} columns.")

    start_time = time.time()

    # 使用K折交叉验证评估模型性能
    score_CatBoost = rmsle_cross_val(train_data, np.log(train_labels), cat_features, 5, model_CatBoost)
    print(f"CatBoostRegressor average score: {score_CatBoost.mean():.4f} ({score_CatBoost.std():.4f})\n")

    elapsed_time = time.time() - start_time
    print(f"Time taken for training: {elapsed_time:.2f} seconds")

    # 最终在整个数据集上训练模型
    train_pool = Pool(train_data, train_labels, cat_features=cat_features)
    model_CatBoost.fit(train_pool, verbose=False)

    # 保存模型
    model_path = 'catboost_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_CatBoost, f)
    print(f"Training Completed. Model Saved at {model_path}")