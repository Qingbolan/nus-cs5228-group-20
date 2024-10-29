import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR  # 添加SVR的导入
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from category_encoders import TargetEncoder
import pickle
import time
import logging
import warnings
import os
from typing import List, Tuple, Optional, Dict, Any

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)

def load_and_preprocess_data(file_path: str) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """加载并初步预处理数据。"""
    data = pd.read_csv(file_path)
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns='Unnamed: 0')
    
    X = data.drop('price', axis=1) if 'price' in data.columns else data
    y = data['price'] if 'price' in data.columns else None
    
    logging.info(f"{file_path} 中的列: {X.columns.tolist()}")
    
    return X, y

def preprocess_features(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    num_imputer: Optional[SimpleImputer] = None,
    cat_imputer: Optional[SimpleImputer] = None, 
    target_encoder: Optional[TargetEncoder] = None, 
    scaler: Optional[StandardScaler] = None, 
    target_encode_cols: List[str] = [], 
    encoding_smoothing: float = 1.0
) -> Tuple[pd.DataFrame, SimpleImputer, SimpleImputer, TargetEncoder, StandardScaler]:
    """对特征进行预处理。"""
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

    # 类别特征填补和编码
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
        
        other_categorical = [col for col in categorical_features if col not in target_encode_features]
        if other_categorical:
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            encoded_features = encoder.fit_transform(X[other_categorical])
            encoded_feature_names = encoder.get_feature_names_out(other_categorical)
            encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=X.index)
            X = pd.concat([X, encoded_df], axis=1)
            X = X.drop(columns=other_categorical)

    return X, num_imputer, cat_imputer, target_encoder, scaler

def find_optimal_clusters(
    X: pd.DataFrame,
    y: pd.Series,
    max_clusters: int = 10,
    features_for_clustering: List[str] = ['depreciation', 'coe', 'dereg_value']
) -> int:
    """使用轮廓系数寻找最佳聚类数量。"""
    logging.info("开始聚类优化")
    
    missing_features = [f for f in features_for_clustering if f not in X.columns]
    if missing_features:
        raise ValueError(f"缺失的聚类特征: {missing_features}")
    
    # 提取用于聚类的特征
    cluster_features_df = X[features_for_clustering]
    imputer = SimpleImputer(strategy='median')
    cluster_features_clean = imputer.fit_transform(cluster_features_df)
    
    # 标准化特征
    scaler = StandardScaler()
    cluster_features_scaled = scaler.fit_transform(cluster_features_clean)
    
    # 将目标变量和特征结合用于聚类
    cluster_features = np.column_stack([np.log1p(y), cluster_features_scaled])
    
    silhouette_scores = []
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(cluster_features)
        silhouette_avg = silhouette_score(cluster_features, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        logging.info(f"n_clusters = {n_clusters}, 轮廓系数 = {silhouette_avg:.4f}")
    
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    logging.info(f"选择的最佳聚类数量: {optimal_clusters}")
    return optimal_clusters

def create_price_clusters(
    X: pd.DataFrame,
    y: pd.Series,
    n_clusters: int,
    features_for_clustering: List[str] = ['depreciation', 'coe', 'dereg_value']
) -> Tuple[KMeans, np.ndarray, pd.DataFrame]:
    """使用KMeans创建基于价格的聚类。"""
    logging.info(f"创建 {n_clusters} 个价格聚类")
    
    cluster_features_df = X[features_for_clustering]
    imputer = SimpleImputer(strategy='median')
    cluster_features_clean = imputer.fit_transform(cluster_features_df)
    
    scaler = StandardScaler()
    cluster_features_scaled = scaler.fit_transform(cluster_features_clean)
    
    # 初始化聚类中心
    price_percentiles = np.percentile(y, np.linspace(0, 100, n_clusters))
    initial_centers = np.column_stack([
        np.log1p(price_percentiles),
        np.percentile(cluster_features_scaled, np.linspace(0, 100, n_clusters), axis=0)
    ])
    
    kmeans = KMeans(n_clusters=n_clusters, init=initial_centers, n_init=10, random_state=42)
    cluster_features = np.column_stack([np.log1p(y), cluster_features_scaled])
    price_clusters = kmeans.fit_predict(cluster_features)
    
    # 聚类统计信息
    cluster_info = []
    for cluster in range(n_clusters):
        cluster_prices = y[price_clusters == cluster]
        cluster_info.append({
            'cluster': cluster,
            'min': cluster_prices.min(),
            'max': cluster_prices.max(),
            'median': cluster_prices.median(),
            'count': len(cluster_prices)
        })
    
    cluster_df = pd.DataFrame(cluster_info)
    logging.info("价格聚类信息:")
    logging.info(cluster_df)
    
    kmeans.feature_imputer = imputer
    kmeans.feature_scaler = scaler
    return kmeans, price_clusters, cluster_df

def predict_cluster(
    X: pd.DataFrame,
    y: Optional[pd.Series],
    kmeans_model: KMeans,
    features_for_clustering: List[str] = ['depreciation', 'coe', 'dereg_value']
) -> np.ndarray:
    """为新数据预测聚类。"""
    cluster_features_df = X[features_for_clustering]
    cluster_features_clean = kmeans_model.feature_imputer.transform(cluster_features_df)
    cluster_features_scaled = kmeans_model.feature_scaler.transform(cluster_features_clean)
    
    dummy_y = np.zeros(len(X)) if y is None else np.log1p(y)
    cluster_features = np.column_stack([dummy_y, cluster_features_scaled])
    
    return kmeans_model.predict(cluster_features)

def train_evaluate_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Dict[str, Any]
) -> lgb.Booster:
    """训练并评估LightGBM模型。"""
    train_data = lgb.Dataset(X_train, label=np.log1p(y_train))
    val_data = lgb.Dataset(X_val, label=np.log1p(y_val), reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        # early_stopping_rounds=50,
        # verbose_eval=False
    )
    
    return model

def train_svr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict[str, Any]
) -> SVR:
    """训练SVR模型。"""
    model = SVR(**params)
    model.fit(X_train, y_train)
    
    val_predictions = model.predict(X_val)
    val_mse = mean_squared_error(y_val, val_predictions)
    val_r2 = r2_score(y_val, val_predictions)
    
    logging.info(f"SVR 验证 RMSE: {np.sqrt(val_mse):.4f}")
    logging.info(f"SVR 验证 R2: {val_r2:.4f}")
    
    return model

def ensemble_with_ridge(
    base_models_oof: pd.DataFrame,
    base_models_test: pd.DataFrame,
    target: pd.Series,
    n_folds: int = 5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[float], List[np.ndarray]]:
    """
    使用Ridge回归集成多个模型
    
    参数:
    --------
    base_models_oof: DataFrame
        基础模型的OOF（Out-Of-Fold）预测
    base_models_test: DataFrame 
        基础模型的测试集预测
    target: Series
        训练数据的目标值
    n_folds: int
        交叉验证的折数
    random_state: int
        随机种子
    
    返回:
    --------
    test_pred: array
        测试集的最终预测
    oof_pred: array
        训练集的OOF预测
    cv_scores: list
        交叉验证得分
    weights: list
        每折Ridge回归的权重
    """
    
    # Ridge回归参数
    ridge_params = {
        "alpha": 1.0,
        "fit_intercept": False,
        "max_iter": 200,
        "tol": 0.0001,
        "solver": 'lbfgs',
        "positive": True,
        "random_state": random_state
    }
    
    # 初始化交叉验证
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    scores = []
    weights = []
    test_pred = np.zeros(len(base_models_test))
    oof_pred = np.zeros(len(base_models_oof))
    
    # 填补缺失值
    imputer = SimpleImputer(strategy='mean')
    base_models_oof_imputed = imputer.fit_transform(base_models_oof)
    base_models_test_imputed = imputer.transform(base_models_test)
    
    # 进行交叉验证训练Ridge回归模型
    for fold, (train_idx, valid_idx) in enumerate(kf.split(base_models_oof_imputed), 1):
        X_train, X_valid = base_models_oof_imputed[train_idx], base_models_oof_imputed[valid_idx]
        y_train, y_valid = target.iloc[train_idx], target.iloc[valid_idx]
        
        model = Ridge(**ridge_params)
        model.fit(X_train, y_train)
        
        # 预测
        valid_pred = model.predict(X_valid)
        test_fold_pred = model.predict(base_models_test_imputed)
        
        # 存储结果
        oof_pred[valid_idx] = valid_pred
        test_pred += test_fold_pred / n_folds
        scores.append(np.sqrt(mean_squared_error(y_valid, valid_pred)))
        weights.append(model.coef_)
    
    logging.info(f"交叉验证 RMSE 分数: {scores}")
    logging.info(f"平均 RMSE: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    logging.info(f"训练集 OOF RMSE: {np.sqrt(mean_squared_error(target, oof_pred)):.4f}")
    
    return test_pred, oof_pred, scores, weights

def post_process_predictions(predictions: np.ndarray, min_price: float = 700, max_price: float = 2900000) -> np.ndarray:
    """后处理预测结果，限制价格范围。"""
    return np.clip(predictions, min_price, max_price)

def main():
    """主执行函数。"""
    np.random.seed(42)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # 加载数据
    X, y = load_and_preprocess_data('preprocessing/2024-10-21-silan/train_cleaned.csv')
    
    logging.info("目标变量（price）统计:")
    logging.info(y.describe())
    
    features_for_clustering = ['depreciation', 'coe', 'dereg_value']
    
    # 找到最佳聚类数量
    optimal_clusters = find_optimal_clusters(X, y, max_clusters=3, features_for_clustering=features_for_clustering)
    
    # 创建价格聚类
    kmeans_model, price_clusters, cluster_info = create_price_clusters(
        X, y, n_clusters=optimal_clusters,
        features_for_clustering=features_for_clustering
    )
    
    # 定义交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 初始化存储结构
    base_models_oof = pd.DataFrame(index=X.index)
    base_models_test = pd.DataFrame()
    cv_scores = []
    feature_importance_list = []
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
    
    # GradientBoostingRegressor 的超参数
    gb_params = {
        'n_estimators': 3000,
        'learning_rate': 0.05,
        'max_depth': 5,
        'min_samples_split': 20,
        'min_samples_leaf': 15,
        'loss': 'huber',
        'random_state': 42
    }
    
    # CatBoostRegressor 的超参数
    cb_params = {
        'iterations': 3000,
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
    
    # SVR 的超参数
    svr_params = {
        'kernel': 'rbf',  
        'C': 100,        # 正则化参数
        'epsilon': 0.1,   # epsilon-tube
        'gamma': 'scale', # RBF核参数
        'cache_size': 2000,  # 缓存大小(MB)
        'verbose': False
    }
    
    # 初始化测试集预测存储
    base_models_test = pd.DataFrame()
    
    start_time = time.time()
    
    for cluster in range(len(cluster_info)):
        logging.info(f"\n训练聚类 {cluster} 的模型")
        X_cluster = X[price_clusters == cluster]
        y_cluster = y[price_clusters == cluster]
        
        cluster_models = []
        
        # 初始化存储该聚类的OOF预测
        oof_preds_cluster = pd.DataFrame(index=X_cluster.index, columns=['lgb', 'gb', 'cb', 'svr'])
        
        for fold, (train_index, val_index) in enumerate(kf.split(X_cluster), 1):
            logging.info(f"聚类 {cluster} - 第 {fold} 折")
            
            X_train, X_val = X_cluster.iloc[train_index].copy(), X_cluster.iloc[val_index].copy()
            y_train, y_val = y_cluster.iloc[train_index], y_cluster.iloc[val_index]
            
            # 预处理特征
            X_train_processed, num_imputer, cat_imputer, target_encoder, scaler = preprocess_features(X_train, y_train)
            X_val_processed, _, _, _, _ = preprocess_features(X_val, y_val, num_imputer, cat_imputer, target_encoder, scaler)
            
            # 训练 LightGBM
            model_lgb = train_evaluate_lightgbm(X_train_processed, y_train, X_val_processed, y_val, lgb_params)
            
            # 训练 GradientBoostingRegressor
            model_gb = GradientBoostingRegressor(**gb_params)
            model_gb.fit(X_train_processed, np.log1p(y_train))
            
            # 训练 CatBoostRegressor
            model_cb = CatBoostRegressor(**cb_params)
            model_cb.fit(X_train_processed, np.log1p(y_train), eval_set=(X_val_processed, np.log1p(y_val)))
            
            # 训练 SVR
            model_svr = train_svr(X_train_processed, np.log1p(y_train), X_val_processed, np.log1p(y_val), svr_params)
            
            # 预测
            preds_lgb = np.expm1(model_lgb.predict(X_val_processed, num_iteration=model_lgb.best_iteration))
            preds_gb = np.expm1(model_gb.predict(X_val_processed))
            preds_cb = np.expm1(model_cb.predict(X_val_processed))
            preds_svr = np.expm1(model_svr.predict(X_val_processed))
            
            # 检查并处理 NaN
            preds_lgb = np.nan_to_num(preds_lgb, nan=np.mean(preds_lgb))
            preds_gb = np.nan_to_num(preds_gb, nan=np.mean(preds_gb))
            preds_cb = np.nan_to_num(preds_cb, nan=np.mean(preds_cb))
            preds_svr = np.nan_to_num(preds_svr, nan=np.mean(preds_svr))
            
            # 存储基础模型的 OOF 预测
            oof_preds_cluster.loc[X_val.index, 'lgb'] = preds_lgb
            oof_preds_cluster.loc[X_val.index, 'gb'] = preds_gb
            oof_preds_cluster.loc[X_val.index, 'cb'] = preds_cb
            oof_preds_cluster.loc[X_val.index, 'svr'] = preds_svr
            
            # 保存 LightGBM 的特征重要性
            importance = model_lgb.feature_importance(importance_type='gain')
            feature_importance = pd.DataFrame({'feature': X_train_processed.columns, 'importance': importance})
            feature_importance_list.append(feature_importance)
            
            # 保存模型及其预处理器
            cluster_models.append({
                'lightgbm': model_lgb,
                'gradient_boosting': model_gb,
                'catboost': model_cb,
                'svr': model_svr,
                'preprocessors': {
                    'num_imputer': num_imputer,
                    'cat_imputer': cat_imputer,
                    'target_encoder': target_encoder,
                    'scaler': scaler
                }
            })
        
        # 存储该聚类的OOF预测
        base_models_oof[f'cluster_{cluster}_lgb'] = oof_preds_cluster['lgb']
        base_models_oof[f'cluster_{cluster}_gb'] = oof_preds_cluster['gb']
        base_models_oof[f'cluster_{cluster}_cb'] = oof_preds_cluster['cb']
        base_models_oof[f'cluster_{cluster}_svr'] = oof_preds_cluster['svr']
        
        # 生成测试集预测
        X_test, _ = load_and_preprocess_data('preprocessing/2024-10-21-silan/test_cleaned.csv')
        test_clusters = predict_cluster(X_test, None, kmeans_model, features_for_clustering)
        X_test_cluster = X_test[test_clusters == cluster]
        
        if len(X_test_cluster) == 0:
            logging.warning(f"测试数据中没有聚类 {cluster} 的样本。跳过此聚类。")
            continue
        
        # 初始化存储测试集的预测
        test_preds_cluster = pd.DataFrame(index=X_test_cluster.index, columns=['lgb', 'gb', 'cb', 'svr'])
        
        for model_dict in cluster_models:
            model_lgb = model_dict['lightgbm']
            model_gb = model_dict['gradient_boosting']
            model_cb = model_dict['catboost']
            model_svr = model_dict['svr']
            preprocessors = model_dict['preprocessors']
            
            try:
                X_test_processed, _, _, _, _ = preprocess_features(X_test_cluster, y=None, **preprocessors)
                
                preds_lgb = np.expm1(model_lgb.predict(X_test_processed, num_iteration=model_lgb.best_iteration))
                preds_gb = np.expm1(model_gb.predict(X_test_processed))
                preds_cb = np.expm1(model_cb.predict(X_test_processed))
                preds_svr = np.expm1(model_svr.predict(X_test_processed))
                
                preds_lgb = np.nan_to_num(preds_lgb, nan=np.mean(preds_lgb))
                preds_gb = np.nan_to_num(preds_gb, nan=np.mean(preds_gb))
                preds_cb = np.nan_to_num(preds_cb, nan=np.mean(preds_cb))
                preds_svr = np.nan_to_num(preds_svr, nan=np.mean(preds_svr))
                
                test_preds_cluster['lgb'] += preds_lgb / kf.n_splits
                test_preds_cluster['gb'] += preds_gb / kf.n_splits
                test_preds_cluster['cb'] += preds_cb / kf.n_splits
                test_preds_cluster['svr'] += preds_svr / kf.n_splits
            except Exception as e:
                logging.error(f"聚类 {cluster} 的模型预测错误: {str(e)}")
                logging.error(f"测试样本形状: {X_test_cluster.shape}")
                logging.error(f"测试样本列: {X_test_cluster.columns.tolist()}")
                continue
        
        # 存储测试集的预测
        for col in test_preds_cluster.columns:
            if col not in base_models_test.columns:
                base_models_test[col] = test_preds_cluster[col]
            else:
                base_models_test[col] += test_preds_cluster[col]
        
        models.append(cluster_models)
    
    elapsed_time = time.time() - start_time
    logging.info(f"\n总训练时间: {elapsed_time/60:.2f} 分钟")
    
    # 对所有基础模型的 OOF 预测和测试集预测进行集成
    final_test_predictions, final_oof_predictions, ensemble_scores, ensemble_weights = ensemble_with_ridge(
        base_models_oof,
        base_models_test,
        y,
        n_folds=5,
        random_state=42
    )
    
    final_test_predictions = post_process_predictions(final_test_predictions)
    final_oof_predictions = post_process_predictions(final_oof_predictions)
    
    # 计算 OOF 的 RMSE 和 R2
    oof_mse_total = mean_squared_error(y, final_oof_predictions)
    oof_r2_total = r2_score(y, final_oof_predictions)
    logging.info(f"训练集 OOF RMSE: {np.sqrt(oof_mse_total):.4f}")
    logging.info(f"训练集 OOF R2: {oof_r2_total:.4f}")
    
    # 计算特征重要性（仅LightGBM）
    feature_importance = pd.concat(feature_importance_list).groupby('feature').mean().sort_values('importance', ascending=False)
    logging.info("\n前10个重要特征:")
    logging.info(feature_importance.head(10))
    
    # 保存模型
    with open('ensemble_clustered_models.pkl', 'wb') as f:
        pickle.dump({
            'models': models,
            'kmeans_model': kmeans_model,
            'cluster_info': cluster_info,
            'ensemble_weights': ensemble_weights
        }, f)
    logging.info("模型和预处理器已保存。")
    
    # 生成提交文件
    submission = pd.DataFrame({
        'Id': range(len(final_test_predictions)),
        'Predicted': np.round(final_test_predictions).astype(int)
    })
    
    submission.to_csv('./submission_ensemble_clustered_optimized.csv', index=False)
    logging.info("预测完成。提交文件已保存为 'submission_ensemble_clustered_optimized.csv'。")
    
    # 预测统计
    logging.info("\n预测统计:")
    logging.info(f"最小值: {final_test_predictions.min()}")
    logging.info(f"最大值: {final_test_predictions.max()}")
    logging.info(f"均值: {final_test_predictions.mean()}")
    logging.info(f"中位数: {np.median(final_test_predictions)}")

if __name__ == '__main__':
    main()