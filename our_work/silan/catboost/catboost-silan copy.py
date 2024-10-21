import sklearn
import scipy
import scipy.stats as stats
from scipy.stats import skew,boxcox_normmax, zscore
from scipy.special import boxcox1p
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer,KNNImputer
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, make_scorer, mean_absolute_error 
from sklearn.model_selection import KFold, RandomizedSearchCV
from mlxtend.regressor import StackingCVRegressor
from multiprocessing import cpu_count
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sns
import joblib  # 添加这行导入
from catboost import CatBoostRegressor

import pickle
import time
import logging
import os  # 导入 os 模块
from catboost import CatBoostRegressor, Pool, cv
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
import optuna
import numpy as np
import pandas as pd
import random
from joblib import Parallel, delayed
from scipy.stats import skew, boxcox_normmax
from scipy.special import boxcox1p  # 确认从正确的模块导入

import xgboost as xgb
from lightgbm import LGBMRegressor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# 设置随机种子以确保可重复性
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    # 如使用其他库（例如 TensorFlow、PyTorch），也应设置其随机种子

set_seed(42)

# 自定义 SkewnessCorrector 变换器
class SkewnessCorrector(BaseEstimator, TransformerMixin):
    def __init__(self, skew_threshold=0.3):
        """
        应用 Box-Cox 变换以纠正偏斜的数值特征的变换器。

        参数:
        - skew_threshold: float, 偏度超过此阈值的特征将被变换。
        """
        self.skew_threshold = skew_threshold
        self.lambda_ = {}
        self.skewed_features_idx = []

    def fit(self, X, y=None):
        """
        通过识别偏斜特征并计算 Box-Cox 参数来拟合变换器。

        参数:
        - X: numpy.ndarray, 输入数据（仅数值特征）。
        - y: 忽略

        返回:
        - self
        """
        # 计算每个特征的偏度
        skew_values = np.abs(skew(X, axis=0))

        # 识别偏度超过阈值的特征
        self.skewed_features_idx = np.where(skew_values > self.skew_threshold)[0]

        # 计算每个偏斜特征的 Box-Cox lambda
        for idx in self.skewed_features_idx:
            # 确保所有值为正
            self.lambda_[idx] = boxcox_normmax(X[:, idx] + 1)

        return self

    def transform(self, X):
        """
        对偏斜特征应用 Box-Cox 变换。

        参数:
        - X: numpy.ndarray, 输入数据（仅数值特征）。

        返回:
        - X_transformed: numpy.ndarray, 变换后的数据。
        """
        X = X.copy()
        for idx in self.skewed_features_idx:
            X[:, idx] = boxcox1p(X[:, idx], self.lambda_[idx])
        return X

# 定义特征列表
categorical_features = ['make', 'model']  # 根据数据集更新
numeric_features = ['curb_weight', 'power', 'engine_cap']  # 根据数据集更新

# 数值特征预处理管道
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('skew_correct', SkewnessCorrector()),
    ('scaler', RobustScaler())
])

# 类别特征预处理管道
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 合并预处理步骤
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# 定义基模型列表
base_models = [
    ('xgb', xgb.XGBRegressor(n_estimators=800, max_depth=6, learning_rate=0.1, min_child_weight=10, n_jobs=4, random_state=42)),
    ('ridge', Ridge(alpha=10)),
    ('lasso', Lasso(alpha=0.001)),
    ('svr', SVR(C=10, gamma=0.0001, epsilon=0.05)),
    ('lgbm', LGBMRegressor(n_estimators=2000, max_depth=6, learning_rate=0.05)),
    ('gbm', GradientBoostingRegressor(n_estimators=2000, max_depth=6, learning_rate=0.05))
]

# 定义元学习器
meta_learner = Ridge(alpha=10)

# 创建 StackingRegressor
stacking_regressor = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=5,
    n_jobs=-1,
    passthrough=True
)

# 创建完整的管道
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('stacking', stacking_regressor)
])

# 定义 CatBoost 超参数优化函数
def optimize_catboost(X, y, cat_features, n_trials=50):
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 1000, 5000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),  # 已更新
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 10),
            'random_seed': 42,
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'verbose': False,
            'thread_count': min(8, os.cpu_count()),  # 已更新，或设置为其他合理值
            'task_type': 'GPU'  # 如果没有 GPU，可改为 'CPU' 或移除此参数
        }
        model = CatBoostRegressor(**params)
        try:
            cv_results = cv(
                params=params,
                pool=Pool(X, y, cat_features=cat_features),
                fold_count=5,
                partition_random_seed=42,
                shuffle=True,
                stratified=False,
                verbose=False
            )
            best_rmse = min(cv_results['test-RMSE-mean'])
            return best_rmse
        except Exception as e:
            logging.error(f"Trial {trial.number} failed with parameters: {params} because of error: {e}")
            return float('inf')  # 赋予失败试验一个很差的分数

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, timeout=3600)

    logging.info(f"Best trial: RMSE={study.best_trial.value:.4f}, Params={study.best_trial.params}")
    return study.best_trial.params

if __name__ == '__main__':
    # 读取数据
    logging.info("Loading data...")
    train_data = pd.read_csv('preprocessing/2024-10-10-silan/train_cleaned.csv')
    train_data = train_data.drop(columns=train_data.columns[0])  # 假设第一列是索引
    train_labels = train_data['price']
    X = train_data.drop(columns=['price'])

    # 拟合 Stacking 模型
    logging.info("Training the stacking model...")
    start_time = time.time()
    model_pipeline.fit(X, train_labels)
    elapsed_time = time.time() - start_time
    logging.info(f"Training completed in {elapsed_time:.2f} seconds.")

    # 进行 CatBoost 超参数优化
    logging.info("Starting hyperparameter optimization for CatBoost...")
    best_params = optimize_catboost(X, np.log(train_labels), categorical_features, n_trials=50)

    # 使用最佳参数训练最终的 CatBoost 模型
    logging.info("Training final CatBoost model with optimized parameters...")
    final_catboost = CatBoostRegressor(**best_params)
    final_pool = Pool(X, train_labels, cat_features=categorical_features)
    final_catboost.fit(final_pool, verbose=False)

    # 保存模型
    logging.info("Saving models...")
    joblib.dump(model_pipeline, 'stacking_model_pipeline.pkl')
    joblib.dump(final_catboost, 'catboost_model.pkl')
    logging.info("Models saved successfully.")