import sklearn
import scipy
import numpy as np
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool, cv
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import skew
import optuna
import random
import logging
import time
import joblib
from functools import partial
from multiprocessing import Pool as ProcessPool

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

class SkewnessCorrector(BaseEstimator, TransformerMixin):
    def __init__(self, skew_threshold=0.5):
        self.skew_threshold = skew_threshold
        self.skewed_features_idx = []

    def fit(self, X, y=None):
        skew_values = np.abs(skew(X, axis=0))
        self.skewed_features_idx = np.where(skew_values > self.skew_threshold)[0]
        return self

    def transform(self, X):
        X = X.copy()
        for idx in self.skewed_features_idx:
            X[:, idx] = np.log1p(X[:, idx])
        return X

categorical_features = ['make', 'model']
numeric_features = ['curb_weight', 'power', 'engine_cap']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('skew_correct', SkewnessCorrector()),
    ('scaler', RobustScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    n_jobs=-1
)

base_models = [
    ('xgb', xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42)),
    ('ridge', Ridge(alpha=10)),
    ('lasso', Lasso(alpha=0.001)),
    ('svr', SVR(C=10, gamma=0.0001, epsilon=0.05)),
    ('lgbm', LGBMRegressor(n_estimators=500, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42)),
    ('gbm', GradientBoostingRegressor(n_estimators=500, max_depth=6, learning_rate=0.1, random_state=42))
]

meta_learner = Ridge(alpha=10)

stacking_regressor = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=3,
    n_jobs=-1,
    passthrough=True
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('stacking', stacking_regressor)
])

def optimize_catboost(trial, X, y, cat_features):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 10),
        'random_seed': 42,
        'loss_function': 'RMSE',  # 改为 RMSE
        'eval_metric': 'RMSE',    # 改为 RMSE
        'verbose': False,
        'task_type': 'GPU'
    }
    try:
        cv_results = cv(
            params=params,
            pool=Pool(X, y, cat_features=cat_features),
            fold_count=3,
            partition_random_seed=42,
            shuffle=True,
            stratified=False,
            verbose=False
        )
        return min(cv_results['test-RMSE-mean'])
    except Exception as e:
        logging.error(f"Error in CatBoost CV: {e}")
        return float('inf')  # 返回一个大值，这样这次尝试会被视为失败

if __name__ == '__main__':
    logging.info("Loading data...")
    train_data = pd.read_csv('preprocessing/2024-10-10-silan/train_cleaned.csv')
    train_data = train_data.drop(columns=train_data.columns[0])
    train_labels = train_data['price']
    X = train_data.drop(columns=['price'])

    logging.info("Training the stacking model...")
    start_time = time.time()
    model_pipeline.fit(X, np.log(train_labels))
    elapsed_time = time.time() - start_time
    logging.info(f"Training completed in {elapsed_time:.2f} seconds.")

    logging.info("Starting hyperparameter optimization for CatBoost...")
    study = optuna.create_study(direction='minimize')
    optimize_func = partial(optimize_catboost, X=X, y=np.log(train_labels), cat_features=categorical_features)
    study.optimize(optimize_func, n_trials=30, timeout=1800)

    logging.info(f"Best trial: RMSE={study.best_trial.value:.4f}, Params={study.best_trial.params}")

    logging.info("Training final CatBoost model with optimized parameters...")
    final_catboost = CatBoostRegressor(**study.best_trial.params)
    final_pool = Pool(X, np.log(train_labels), cat_features=categorical_features)
    final_catboost.fit(final_pool, verbose=False)

    logging.info("Saving models...")
    joblib.dump(model_pipeline, 'stacking_model_pipeline.pkl')
    joblib.dump(final_catboost, 'catboost_model.pkl')
    logging.info("Models saved successfully.")