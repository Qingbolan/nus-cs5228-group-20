import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import TargetEncoder
import pickle
import time
import logging
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data(file_path):
    """
    加载并预处理数据
    """
    data = pd.read_csv(file_path)
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns='Unnamed: 0')
    
    X = data.drop('price', axis=1) if 'price' in data.columns else data
    y = data['price'] if 'price' in data.columns else None
    
    logging.info(f"Columns in {file_path}: {X.columns}")
    
    return X, y

def preprocess_features(X, y=None, num_imputer=None, cat_imputer=None, 
                        target_encoder=None, scaler=None, 
                        target_encode_cols=[], 
                        encoding_smoothing=1.0):
    """
    特征预处理，包括缺失值填补、标准化、编码等
    """
    X = X.copy()
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    columns_to_standardize = ['curb_weight', 'power', 'engine_cap', 'depreciation']
    columns_to_standardize = [col for col in columns_to_standardize if col in X.columns]
    
    # 数值特征填补
    if num_imputer is None:
        num_imputer = SimpleImputer(strategy='median')
        X[numeric_features] = pd.DataFrame(num_imputer.fit_transform(X[numeric_features]), 
                                           columns=numeric_features, 
                                           index=X.index)
    else:
        X[numeric_features] = pd.DataFrame(num_imputer.transform(X[numeric_features]), 
                                           columns=numeric_features, 
                                           index=X.index)
    
    # 数值特征标准化
    if columns_to_standardize:
        if scaler is None:
            scaler = StandardScaler()
            X[columns_to_standardize] = pd.DataFrame(scaler.fit_transform(X[columns_to_standardize]), 
                                                     columns=columns_to_standardize, 
                                                     index=X.index)
        else:
            X[columns_to_standardize] = pd.DataFrame(scaler.transform(X[columns_to_standardize]), 
                                                     columns=columns_to_standardize, 
                                                     index=X.index)

    # 分类特征处理
    if categorical_features:
        if cat_imputer is None:
            cat_imputer = SimpleImputer(strategy='constant', fill_value='unknown')
            X[categorical_features] = pd.DataFrame(cat_imputer.fit_transform(X[categorical_features]), 
                                                   columns=categorical_features, 
                                                   index=X.index)
        else:
            X[categorical_features] = pd.DataFrame(cat_imputer.transform(X[categorical_features]), 
                                                   columns=categorical_features, 
                                                   index=X.index)
        
        # 目标编码
        target_encode_features = [col for col in target_encode_cols if col in categorical_features]
        
        if target_encode_features:
            if target_encoder is None:
                target_encoder = TargetEncoder(cols=target_encode_features, smoothing=encoding_smoothing)
                X[target_encode_features] = pd.DataFrame(target_encoder.fit_transform(X[target_encode_features], y), 
                                                         columns=target_encode_features, 
                                                         index=X.index)
            else:
                X[target_encode_features] = pd.DataFrame(target_encoder.transform(X[target_encode_features]), 
                                                         columns=target_encode_features, 
                                                         index=X.index)
        
        # 独热编码
        other_categorical = [col for col in categorical_features if col not in target_encode_features]
        if other_categorical:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_features = encoder.fit_transform(X[other_categorical])
            encoded_feature_names = encoder.get_feature_names_out(other_categorical)
            encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=X.index)
            X = pd.concat([X, encoded_df], axis=1)
            X = X.drop(columns=other_categorical)

    return X, num_imputer, cat_imputer, target_encoder, scaler

def train_evaluate_lightgbm(X_train, y_train, X_val, y_val, params):
    """
    训练 LightGBM 模型
    """
    train_data = lgb.Dataset(X_train, label=np.log1p(y_train))
    val_data = lgb.Dataset(X_val, label=np.log1p(y_val), reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        # early_stopping_rounds=100,
        # verbose_eval=False  # 关闭训练过程的输出
    )
    
    return model

def train_evaluate_gb(X_train, y_train, X_val, y_val, params):
    """
    训练 GradientBoostingRegressor 模型
    """
    model = GradientBoostingRegressor(**params)
    model.fit(X_train, np.log1p(y_train))
    return model

def train_evaluate_catboost(X_train, y_train, X_val, y_val, params):
    """
    训练 CatBoostRegressor 模型
    """
    model = CatBoostRegressor(**params)
    model.fit(X_train, np.log1p(y_train),
              eval_set=(X_val, np.log1p(y_val)),
            #   verbose=False  # 关闭训练过程的输出
              )
    return model

def post_process_predictions(predictions, min_price=700, max_price=2000000):
    """
    后处理预测结果，限制在合理的价格范围内
    """
    return np.clip(predictions, min_price, max_price)

def main():
    np.random.seed(42)
    
    # 加载训练数据
    X, y = load_and_preprocess_data('preprocessing/2024-10-21-silan/train_cleaned.csv')
    
    logging.info("Target variable (price) statistics:")
    logging.info(y.describe())
    
    features_for_clustering = ['depreciation', 'coe', 'dereg_value']
    
    # 预处理所有特征
    X_processed, num_imputer, cat_imputer, target_encoder, scaler = preprocess_features(
        X, y, target_encode_cols=features_for_clustering
    )
    
    # 设置交叉验证的fold数量为10
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # 初始化Oof_predictions
    oof_predictions = np.zeros(len(X))
    oof_mse = []
    oof_r2 = []
    models = []
    
    # 定义各模型的超参数
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': 42,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'max_depth': -1,
        'min_child_samples': 20,
        'cat_smooth': 10,
        'cat_l2': 10,
    }
    
    gb_params = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 5,
        'min_samples_split': 20,
        'min_samples_leaf': 15,
        'loss': 'huber',
        'random_state': 42
    }
    
    cb_params = {
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 10,
        'min_data_in_leaf': 20,
        'random_strength': 0.5,
        'bagging_temperature': 0.2,
        'od_type': 'Iter',
        'od_wait': 50,
        'random_seed': 42,
        'verbose': False
    }
    
    start_time = time.time()
    
    # 使用 tqdm 显示训练进度
    for fold, (train_index, val_index) in enumerate(tqdm(kf.split(X), desc="Cross-Validation Folds")):
        logging.info(f"\nStarting Fold {fold + 1}")
        
        X_train, X_val = X_processed.iloc[train_index], X_processed.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # 训练 LightGBM
        model_lgb = train_evaluate_lightgbm(X_train, y_train, X_val, y_val, lgb_params)
        
        # 训练 GradientBoostingRegressor
        model_gb = train_evaluate_gb(X_train, y_train, X_val, y_val, gb_params)
        
        # 训练 CatBoostRegressor
        model_cb = train_evaluate_catboost(X_train, y_train, X_val, y_val, cb_params)
        
        # 预测
        preds_lgb = np.expm1(model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration))
        preds_gb = np.expm1(model_gb.predict(X_val))
        preds_cb = np.expm1(model_cb.predict(X_val))
        
        # 集成预测（简单平均）
        preds_ensemble = (preds_lgb + preds_gb + preds_cb) / 3.0
        
        # 记录OOF预测
        oof_predictions[val_index] = preds_ensemble
        
        # 记录指标
        rmse = np.sqrt(mean_squared_error(y_val, preds_ensemble))
        r2 = r2_score(y_val, preds_ensemble)
        oof_mse.append(rmse ** 2)
        oof_r2.append(r2)
        logging.info(f"Fold {fold + 1} RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
        # 保存模型
        models.append({
            'lightgbm': model_lgb,
            'gradient_boosting': model_gb,
            'catboost': model_cb,
            'preprocessors': {
                'num_imputer': num_imputer,
                'cat_imputer': cat_imputer,
                'target_encoder': target_encoder,
                'scaler': scaler
            }
        })
    
    elapsed_time = time.time() - start_time
    logging.info(f"\nTotal training time: {elapsed_time / 60:.2f} minutes")
    
    # 计算整体的Oof RMSE和R2
    oof_predictions = post_process_predictions(oof_predictions)
    oof_mse_total = mean_squared_error(y, oof_predictions)
    oof_r2_total = r2_score(y, oof_predictions)
    logging.info(f"Out-of-fold RMSE: {np.sqrt(oof_mse_total):.4f}")
    logging.info(f"Out-of-fold R2: {oof_r2_total:.4f}")
    
    # 保存模型及预处理器
    with open('ensemble_models.pkl', 'wb') as f:
        pickle.dump({
            'models': models,
            'preprocessors': {
                'num_imputer': num_imputer,
                'cat_imputer': cat_imputer,
                'target_encoder': target_encoder,
                'scaler': scaler
            }
        }, f)
    logging.info("Models and preprocessors saved as 'ensemble_models.pkl'.")
    
    # 训练全量模型
    logging.info("\nTraining final models on the entire dataset")
    
    # 预处理训练数据全量
    X_full_train_processed, _, _, _, _ = preprocess_features(X, y, num_imputer, cat_imputer, target_encoder, scaler, target_encode_cols=features_for_clustering)
    
    # 训练 LightGBM
    final_model_lgb = train_evaluate_lightgbm(X_full_train_processed, y, X_full_train_processed, y, lgb_params)
    
    # 训练 GradientBoostingRegressor
    final_model_gb = train_evaluate_gb(X_full_train_processed, y, X_full_train_processed, y, gb_params)
    
    # 训练 CatBoostRegressor
    final_model_cb = train_evaluate_catboost(X_full_train_processed, y, X_full_train_processed, y, cb_params)
    
    # 保存最终模型
    final_models = {
        'lightgbm': final_model_lgb,
        'gradient_boosting': final_model_gb,
        'catboost': final_model_cb,
        'preprocessors': {
            'num_imputer': num_imputer,
            'cat_imputer': cat_imputer,
            'target_encoder': target_encoder,
            'scaler': scaler
        }
    }
    
    with open('final_ensemble_models.pkl', 'wb') as f:
        pickle.dump(final_models, f)
    logging.info("Final models saved as 'final_ensemble_models.pkl'.")
    
    # 预测测试数据
    logging.info("\nPredicting on test data")
    X_test, _ = load_and_preprocess_data('preprocessing/2024-10-21-silan/test_cleaned.csv')
    
    # 预处理测试数据
    X_test_processed, _, _, _, _ = preprocess_features(X_test, y=None, 
                                                        num_imputer=num_imputer, 
                                                        cat_imputer=cat_imputer, 
                                                        target_encoder=target_encoder, 
                                                        scaler=scaler, 
                                                        target_encode_cols=features_for_clustering)
    
    # 预测
    preds_lgb_test = np.expm1(final_model_lgb.predict(X_test_processed, num_iteration=final_model_lgb.best_iteration))
    preds_gb_test = np.expm1(final_model_gb.predict(X_test_processed))
    preds_cb_test = np.expm1(final_model_cb.predict(X_test_processed))
    
    # 集成预测（简单平均）
    final_predictions = (preds_lgb_test + preds_gb_test + preds_cb_test) / 3.0
    
    final_predictions = post_process_predictions(final_predictions)
    
    # 生成提交文件
    submission = pd.DataFrame({
        'Id': X_test.index,
        'Predicted': np.round(final_predictions).astype(int)
    })
    
    submission.to_csv('./1026_ensemble_models_prediction.csv', index=False)
    logging.info("Predictions complete. Submission file saved as '1026_ensemble_models_prediction.csv'.")
    
    # 输出预测统计信息
    logging.info("\nPrediction statistics:")
    logging.info(f"Minimum: {final_predictions.min()}")
    logging.info(f"Maximum: {final_predictions.max()}")
    logging.info(f"Mean: {final_predictions.mean()}")
    logging.info(f"Median: {np.median(final_predictions)}")
    
    elapsed_time_total = time.time() - start_time
    logging.info(f"\nTotal elapsed time: {elapsed_time_total / 60:.2f} minutes")

if __name__ == '__main__':
    main()