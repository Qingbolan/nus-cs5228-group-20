import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
import pickle
import time
from category_encoders import TargetEncoder
import logging
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class MLPRegressor(nn.Module):
    """PyTorch神经网络模型"""
    def __init__(self, input_dim):
        super(MLPRegressor, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(input_dim)
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x = self.batch_norm1(x)
        return self.network(x)

class WeightedEnsembleL2:
    def __init__(self):
        self.weights = None
        
    def fit(self, predictions, y_true):
        """
        使用L2正则化的最小二乘法来优化权重
        predictions: shape (n_samples, n_models)
        y_true: shape (n_samples,)
        """
        n_models = predictions.shape[1]
        
        # 添加正则化项的解析解
        lambda_reg = 0.1  # L2正则化参数
        A = predictions.T @ predictions + lambda_reg * np.eye(n_models)
        b = predictions.T @ y_true
        
        try:
            self.weights = np.linalg.solve(A, b)
            # 归一化权重
            self.weights = np.maximum(0, self.weights)  # 确保权重非负
            self.weights /= np.sum(self.weights)
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用均等权重
            self.weights = np.ones(n_models) / n_models
            
        return self
        
    def predict(self, predictions):
        """
        使用学习到的权重进行预测
        predictions: shape (n_samples, n_models)
        """
        return predictions @ self.weights

class RobustWeightedEnsemble(WeightedEnsembleL2):
    def fit(self, predictions, y_true):
        """改进的集成策略"""
        # 计算各模型的基础性能指标
        base_metrics = []
        for i in range(predictions.shape[1]):
            pred = predictions[:, i]
            rmse = np.sqrt(mean_squared_error(y_true, pred))
            r2 = r2_score(y_true, pred)
            base_metrics.append((rmse, r2))
        
        # 移除性能特别差的模型
        valid_indices = []
        valid_preds = []
        for i, (rmse, r2) in enumerate(base_metrics):
            if r2 > -0.5:  # 设置一个基础阈值
                valid_indices.append(i)
                valid_preds.append(predictions[:, i])
        
        if len(valid_preds) == 0:
            # 如果所有模型都表现很差，使用简单平均
            self.weights = np.ones(predictions.shape[1]) / predictions.shape[1]
        else:
            # 使用有效模型进行优化
            valid_preds = np.column_stack(valid_preds)
            super().fit(valid_preds, y_true)
            
            # 将权重分配回原始模型
            full_weights = np.zeros(predictions.shape[1])
            for i, idx in enumerate(valid_indices):
                full_weights[idx] = self.weights[i]
            self.weights = full_weights
        
        return self

def optimize_svr_params(price_range: str) -> Dict[str, Any]:
    """为不同价格范围优化SVR参数"""
    if price_range == 'low':
        return {
            'kernel': 'rbf',
            'C': 150,        # 增加C值以更好拟合低价区间
            'epsilon': 0.05,  # 降低epsilon以提高精度
            'gamma': 'scale',
            'cache_size': 2000,
        }
    elif price_range == 'medium':
        return {
            'kernel': 'rbf',
            'C': 100,
            'epsilon': 0.1,
            'gamma': 'scale',
            'cache_size': 2000,
        }
    else:  # high
        return {
            'kernel': 'rbf',
            'C': 100,         # 降低C值以避免高价区间过拟合
            'epsilon': 0.05,  # 增加epsilon以提高泛化能力
            'gamma': 'scale',
            'cache_size': 4000,
        }

def determine_price_range(prices: np.ndarray) -> str:
    """根据价格确定范围"""
    median_price = np.median(prices)
    if median_price < 50000:
        return 'low'
    elif median_price < 1500000:
        return 'medium'
    else:
        return 'high'
    
def load_and_preprocess_data(file_path):
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
    X = X.copy()
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    columns_to_standardize = ['curb_weight', 'power', 'engine_cap', 'depreciation']
    columns_to_standardize = [col for col in columns_to_standardize if col in X.columns]
    
    if num_imputer is None:
        num_imputer = SimpleImputer(strategy='median')
        X[numeric_features] = pd.DataFrame(num_imputer.fit_transform(X[numeric_features]), 
                                           columns=numeric_features, 
                                           index=X.index)
    else:
        X[numeric_features] = pd.DataFrame(num_imputer.transform(X[numeric_features]), 
                                           columns=numeric_features, 
                                           index=X.index)
    
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

    if len(categorical_features) > 0:
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
        if len(other_categorical) > 0:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_features = encoder.fit_transform(X[other_categorical])
            encoded_feature_names = encoder.get_feature_names_out(other_categorical)
            encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=X.index)
            X = pd.concat([X, encoded_df], axis=1)
            X = X.drop(columns=other_categorical)

    return X, num_imputer, cat_imputer, target_encoder, scaler

def create_price_clusters(X, y, n_clusters, features_for_clustering=['depreciation', 'coe', 'dereg_value']):
    price_percentiles = np.percentile(y, np.linspace(0, 100, n_clusters))
    initial_centers = np.column_stack([
        np.log1p(price_percentiles),
        np.percentile(X[features_for_clustering], np.linspace(0, 100, n_clusters), axis=0)
    ])

    kmeans = KMeans(n_clusters=n_clusters, init=initial_centers, n_init=10, random_state=42)
    cluster_features = np.column_stack([np.log1p(y), X[features_for_clustering]])
    price_clusters = kmeans.fit_predict(cluster_features)
    
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
    logging.info("Price Cluster Information:")
    logging.info(cluster_df)
    
    return kmeans, price_clusters, cluster_df

def train_evaluate_pytorch(X_train, y_train, X_val, y_val, params):
    """
    训练PyTorch神经网络模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_train.shape[1]
    
    # 将DataFrame转换为numpy数组，再转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train.values).to(device)  # 添加.values
    y_train_tensor = torch.FloatTensor(np.log1p(y_train.values if isinstance(y_train, pd.Series) else y_train)).reshape(-1, 1).to(device)
    X_val_tensor = torch.FloatTensor(X_val.values).to(device)  # 添加.values
    y_val_tensor = torch.FloatTensor(np.log1p(y_val.values if isinstance(y_val, pd.Series) else y_val)).reshape(-1, 1).to(device)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=params.get('batch_size', 128),
        shuffle=True
    )
    
    # 初始化模型
    model = MLPRegressor(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), 
                          lr=params.get('learning_rate', 0.001),
                          weight_decay=params.get('weight_decay', 1e-5))
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
    
    # 训练模型
    epochs = params.get('epochs', 100)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # 验证
        model.eval()
        with torch.no_grad():
            val_output = model(X_val_tensor)
            val_loss = criterion(val_output, y_val_tensor)
            
        scheduler.step(val_loss)
        early_stopping(val_loss)
        
        if early_stopping.early_stop:
            logging.info(f"Early stopping triggered at epoch {epoch}")
            break
    
    return model

def predict_pytorch(model, X):
    """
    使用PyTorch模型进行预测
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        # 将DataFrame转换为numpy数组，再转换为PyTorch张量
        X_tensor = torch.FloatTensor(X.values).to(device)  # 添加.values
        predictions = model(X_tensor).cpu().numpy()
    return predictions.reshape(-1)

def train_evaluate_lightgbm(X_train, y_train, X_val, y_val, params):
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

def train_evaluate_svr(X_train, y_train, price_range=None):
    """
    训练SVR模型，使用优化后的参数
    """
    if price_range is None:
        price_range = determine_price_range(y_train)
    
    params = optimize_svr_params(price_range)
    logging.info(f"Using SVR parameters for {price_range} price range: {params}")
    
    model = SVR(**params)
    model.fit(X_train, np.log1p(y_train))
    return model

def train_evaluate_catboost(X_train, y_train, X_val, y_val, base_predictions, params):
    """
    使用其他模型的预测结果训练加权的CatBoost
    """
    # 计算初始的ensemble weights用于样本加权
    ensemble = RobustWeightedEnsemble()
    ensemble.fit(base_predictions, y_train)
    initial_weights = ensemble.weights
    
    # 计算加权预测结果
    weighted_preds = base_predictions @ initial_weights
    
    # 基于预测误差计算样本权重
    prediction_errors = np.abs(weighted_preds - y_train)
    sample_weights = 1 / (1 + prediction_errors)
    sample_weights = sample_weights / sample_weights.mean()  # 归一化权重
    
    # 微调CatBoost参数
    adjusted_params = params.copy()
    adjusted_params.update({
        'iterations': 3000,
        'learning_rate': 0.05,
        'depth': 6,
        'min_data_in_leaf': 20,
        'l2_leaf_reg': 1.5,  # 增加L2正则化
        'random_strength': 0.8,  # 增加随机性
        'bagging_temperature': 0.5,
        'grow_policy': 'Lossguide',
        'od_type': 'Iter',
        'od_wait': 50
    })
    
    # 训练模型
    model = CatBoostRegressor(**adjusted_params)
    model.fit(
        X_train, 
        np.log1p(y_train),
        sample_weight=sample_weights,
        eval_set=(X_val, np.log1p(y_val)),
        verbose=False
    )
    
    return model, initial_weights

def find_optimal_clusters(X, y, max_clusters=3, features_for_clustering=['depreciation', 'coe', 'dereg_value']):
    from sklearn.metrics import silhouette_score
    cluster_features = np.column_stack([np.log1p(y), X[features_for_clustering]])
    silhouette_scores = []

    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(cluster_features)
        silhouette_avg = silhouette_score(cluster_features, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        logging.info(f"For n_clusters = {n_clusters}, the average silhouette score is : {silhouette_avg}")

    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    logging.info(f"Optimal number of clusters: {optimal_clusters}")
    return optimal_clusters

def predict_cluster(X, y, kmeans_model, preprocessors, features_for_clustering=['depreciation', 'coe', 'dereg_value']):
    X_processed, _, _, _, _ = preprocess_features(X, y, **preprocessors)
    cluster_features = np.column_stack([np.log1p(y) if y is not None else np.zeros(len(X)), X_processed[features_for_clustering]])
    return kmeans_model.predict(cluster_features)

def post_process_predictions(predictions, min_price=700, max_price=2900000):
    return np.clip(predictions, min_price, max_price)

def main():
    np.random.seed(42)
    torch.manual_seed(42)  # 设置PyTorch随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    X, y = load_and_preprocess_data('l2_train_163.csv')
    
    logging.info("Target variable (price) statistics:")
    logging.info(y.describe())
    
    features_for_clustering = ['depreciation', 'coe', 'dereg_value']
    
    # 找到最佳聚类数量
    optimal_clusters = find_optimal_clusters(X, y, max_clusters=3, features_for_clustering=features_for_clustering)
    kmeans_model, price_clusters, cluster_info = create_price_clusters(X, y, n_clusters=optimal_clusters, features_for_clustering=features_for_clustering)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_predictions = np.zeros(len(X))
    oof_mse = []
    oof_r2 = []
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
    
    # XGBoost的超参数
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.05,
        'min_child_weight': 15,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'seed': 42
    }
    
    # GradientBoostingRegressor的超参数
    gb_params = {
        'n_estimators': 3000,
        'learning_rate': 0.05,
        'max_depth': 5,
        'min_samples_split': 20,
        'min_samples_leaf': 15,
        'subsample': 0.8,
        'loss': 'huber',
        'random_state': 42
    }
    
    # CatBoostRegressor的超参数
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
    
    # PyTorch的超参数
    pytorch_params = {
        'batch_size': 128,
        'learning_rate': 0.001,
        'epochs': 100,
        'weight_decay': 1e-5
    }
    
    start_time = time.time()
    
    for cluster in range(len(cluster_info)):
        logging.info(f"\nTraining models for Cluster {cluster}")
        X_cluster = X[price_clusters == cluster]
        y_cluster = y[price_clusters == cluster]
        
        # 确定该聚类的价格范围
        cluster_price_range = determine_price_range(y_cluster)
        logging.info(f"Cluster {cluster} identified as {cluster_price_range} price range")
        
        cluster_models = []
        
        for fold, (train_index, val_index) in enumerate(kf.split(X_cluster), 1):
            logging.info(f"Cluster {cluster} - Fold {fold}")
            
            X_train, X_val = X_cluster.iloc[train_index].copy(), X_cluster.iloc[val_index].copy()
            y_train, y_val = y_cluster.iloc[train_index], y_cluster.iloc[val_index]
            
            X_train_processed, num_imputer, cat_imputer, target_encoder, scaler = preprocess_features(X_train, y_train)
            X_val_processed, _, _, _, _ = preprocess_features(X_val, y_val, num_imputer, cat_imputer, target_encoder, scaler)
            
            # 训练基础模型
            model_lgb = train_evaluate_lightgbm(X_train_processed, y_train, X_val_processed, y_val, lgb_params)
            model_gb = GradientBoostingRegressor(**gb_params)
            model_gb.fit(X_train_processed, np.log1p(y_train))
            
            # 训练XGBoost
            dtrain = xgb.DMatrix(X_train_processed, label=np.log1p(y_train))
            dval = xgb.DMatrix(X_val_processed, label=np.log1p(y_val))
            model_xgb = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=1000,
                evals=[(dtrain, 'train'), (dval, 'valid')],
                # early_stopping_rounds=50,
                # verbose_eval=False
            )
            
            # 条件训练 SVR 和 PyTorch 模型
            if cluster_price_range != 'high':
                # 训练优化后的SVR
                model_svr = train_evaluate_svr(X_train_processed, y_train, cluster_price_range)
                
                # 训练PyTorch模型
                model_pytorch = train_evaluate_pytorch(X_train_processed, y_train, X_val_processed, y_val, pytorch_params)
            else:
                model_svr = None
                model_pytorch = None
                logging.info("Skipping SVR and PyTorch models for high price range cluster.")
            
            # 获取基础模型预测
            preds_lgb = np.expm1(model_lgb.predict(X_train_processed))
            preds_gb = np.expm1(model_gb.predict(X_train_processed))
            preds_xgb = np.expm1(model_xgb.predict(xgb.DMatrix(X_train_processed)))
            
            base_predictions = [preds_lgb, preds_gb, preds_xgb]
            
            if model_svr is not None:
                preds_svr = np.expm1(model_svr.predict(X_train_processed))
                base_predictions.append(preds_svr)
            if model_pytorch is not None:
                preds_pytorch = np.expm1(predict_pytorch(model_pytorch, X_train_processed))
                base_predictions.append(preds_pytorch)
            
            # 组合预测结果用于CatBoost训练
            if len(base_predictions) > 0:
                base_predictions = np.column_stack(base_predictions)
            else:
                logging.warning("No base predictions available for CatBoost training.")
                base_predictions = None
            
            if base_predictions is not None:
                # 训练CatBoost
                model_cb, initial_weights = train_evaluate_catboost(
                    X_train_processed, 
                    y_train, 
                    X_val_processed, 
                    y_val, 
                    base_predictions,
                    cb_params
                )
            else:
                model_cb = None
                initial_weights = None
                logging.warning("CatBoost model was not trained due to lack of base predictions.")
            
            # 验证集预测
            preds_lgb_val = np.expm1(model_lgb.predict(X_val_processed))
            preds_gb_val = np.expm1(model_gb.predict(X_val_processed))
            preds_xgb_val = np.expm1(model_xgb.predict(xgb.DMatrix(X_val_processed)))
            
            base_predictions_val = [preds_lgb_val, preds_gb_val, preds_xgb_val]
            
            if model_svr is not None:
                preds_svr_val = np.expm1(model_svr.predict(X_val_processed))
                base_predictions_val.append(preds_svr_val)
            if model_pytorch is not None:
                preds_pytorch_val = np.expm1(predict_pytorch(model_pytorch, X_val_processed))
                base_predictions_val.append(preds_pytorch_val)
            
            if len(base_predictions_val) > 0:
                base_predictions_val = np.column_stack(base_predictions_val)
            else:
                base_predictions_val = None
            
            if model_cb is not None:
                preds_cb_val = np.expm1(model_cb.predict(X_val_processed))
                # 将 CatBoost 预测添加到堆叠预测中
                predictions_stack = np.column_stack([
                    preds_lgb_val, preds_gb_val, preds_xgb_val
                ])
                if model_svr is not None:
                    predictions_stack = np.column_stack([predictions_stack, preds_svr_val])
                if model_pytorch is not None:
                    predictions_stack = np.column_stack([predictions_stack, preds_pytorch_val])
                predictions_stack = np.column_stack([predictions_stack, preds_cb_val])
            else:
                predictions_stack = None
                logging.warning("CatBoost predictions are not available for stacking.")
            
            if predictions_stack is not None:
                # 最终集成
                final_ensemble = WeightedEnsembleL2()
                final_ensemble.fit(predictions_stack, y_val)
                
                preds_ensemble = final_ensemble.predict(predictions_stack)
                oof_predictions[val_index] = preds_ensemble
                
                rmse = np.sqrt(mean_squared_error(y_val, preds_ensemble))
                r2 = r2_score(y_val, preds_ensemble)
                oof_mse.append(rmse ** 2)
                oof_r2.append(r2)
                logging.info(f"Cluster {cluster} - Fold {fold} RMSE: {rmse:.4f}, R2: {r2:.4f}")
                if initial_weights is not None:
                    logging.info(f"Initial weights: {initial_weights}")
                if final_ensemble.weights is not None:
                    logging.info(f"Final ensemble weights: {final_ensemble.weights}")
            else:
                logging.warning("Final ensemble predictions could not be made due to missing predictions.")
            
            # 保存LightGBM的特征重要性
            importance = model_lgb.feature_importance(importance_type='gain')
            feature_importance = pd.DataFrame({'feature': X_train_processed.columns, 'importance': importance})
            feature_importance_list.append(feature_importance)
            
            # 保存模型
            cluster_models.append({
                'lightgbm': model_lgb,
                'xgboost': model_xgb,
                'gradient_boosting': model_gb,
                'svr': model_svr,
                'pytorch': model_pytorch,
                'catboost': model_cb,
                'ensemble_weights': final_ensemble.weights if 'final_ensemble' in locals() else None,
                'preprocessors': {
                    'num_imputer': num_imputer,
                    'cat_imputer': cat_imputer,
                    'target_encoder': target_encoder,
                    'scaler': scaler
                }
            })
        
        models.append(cluster_models)
    
    elapsed_time = time.time() - start_time
    logging.info(f"\nTotal training time: {elapsed_time/60:.2f} minutes")
    
    oof_predictions = post_process_predictions(oof_predictions)
    oof_mse_total = mean_squared_error(y, oof_predictions)
    oof_r2_total = r2_score(y, oof_predictions)
    logging.info(f"Out-of-fold RMSE: {np.sqrt(oof_mse_total):.4f}")
    logging.info(f"Out-of-fold R2: {oof_r2_total:.4f}")
    
    feature_importance = pd.concat(feature_importance_list).groupby('feature').mean().sort_values('importance', ascending=False)
    logging.info("\nTop 10 important features:")
    logging.info(feature_importance.head(10))
    
    with open('ensemble_clustered_models_new2.pkl', 'wb') as f:
        pickle.dump({
            'models': models,
            'kmeans_model': kmeans_model,
            'cluster_info': cluster_info
        }, f)
    logging.info("Models and preprocessors saved.")
    
    # 预测测试数据
    X_test, _ = load_and_preprocess_data('l2_test_163.csv')
    
    dummy_y_test = np.zeros(len(X_test))
    test_clusters = predict_cluster(X_test, dummy_y_test, kmeans_model, models[0][0]['preprocessors'], features_for_clustering)
    
    final_predictions = np.zeros(len(X_test))
    prediction_counts = np.zeros(len(X_test))
    
    for cluster in range(len(cluster_info)):
        logging.info(f"Predicting for Cluster {cluster}")
        cluster_mask = test_clusters == cluster
        X_test_cluster = X_test[cluster_mask]
        
        if len(X_test_cluster) == 0:
            logging.warning(f"No samples in test data for cluster {cluster}. Skipping this cluster.")
            continue
        
        cluster_models = models[cluster]
        cluster_predictions = np.zeros(len(X_test_cluster))
        num_models = 0
        
        for model_dict in cluster_models:
            model_lgb = model_dict['lightgbm']
            model_xgb = model_dict['xgboost']
            model_gb = model_dict['gradient_boosting']
            model_svr = model_dict['svr']
            model_pytorch = model_dict['pytorch']
            model_cb = model_dict['catboost']
            ensemble_weights = model_dict['ensemble_weights']
            preprocessors = model_dict['preprocessors']
            
            try:
                X_test_processed, _, _, _, _ = preprocess_features(X_test_cluster, y=None, **preprocessors)
                
                preds_lgb = np.expm1(model_lgb.predict(X_test_processed))
                preds_gb = np.expm1(model_gb.predict(X_test_processed))
                preds_xgb = np.expm1(model_xgb.predict(xgb.DMatrix(X_test_processed)))
                
                base_predictions_test = [preds_lgb, preds_gb, preds_xgb]
                
                if model_svr is not None:
                    preds_svr = np.expm1(model_svr.predict(X_test_processed))
                    base_predictions_test.append(preds_svr)
                if model_pytorch is not None:
                    preds_pytorch = np.expm1(predict_pytorch(model_pytorch, X_test_processed))
                    base_predictions_test.append(preds_pytorch)
                
                if model_cb is not None:
                    preds_cb = np.expm1(model_cb.predict(X_test_processed))
                    base_predictions_test.append(preds_cb)
                
                if len(base_predictions_test) > 0:
                    base_predictions_test = np.column_stack(base_predictions_test)
                else:
                    logging.warning(f"No base predictions available for cluster {cluster} during testing.")
                    continue
                
                # 使用集成权重进行最终预测
                if ensemble_weights is not None:
                    preds = base_predictions_test @ ensemble_weights
                    cluster_predictions += preds
                    num_models += 1
                else:
                    logging.warning(f"Ensemble weights not available for cluster {cluster}. Skipping predictions.")
                    continue
            except Exception as e:
                logging.error(f"Error predicting for cluster {cluster} with one of the models: {str(e)}")
                logging.error(f"Shape of X_test_cluster: {X_test_cluster.shape}")
                logging.error(f"Columns in X_test_cluster: {X_test_cluster.columns}")
                continue
        
        if num_models > 0:
            final_predictions[cluster_mask] += cluster_predictions
            prediction_counts[cluster_mask] += num_models
        else:
            logging.warning(f"No successful model predictions for cluster {cluster}.")
    
    # 避免除以零
    prediction_counts[prediction_counts == 0] = 1
    final_predictions /= prediction_counts
    
    final_predictions = post_process_predictions(final_predictions)
    
    submission = pd.DataFrame({
        'Id': range(len(final_predictions)),
        'Predicted': np.round(final_predictions).astype(int)
    })
    
    submission.to_csv('./submission_ensemble_clustered_optimized.csv', index=False)
    logging.info("Predictions complete. Submission file saved as 'submission_ensemble_clustered_optimized.csv'.")
    
    logging.info("\nPrediction statistics:")
    logging.info(f"Minimum: {final_predictions.min()}")
    logging.info(f"Maximum: {final_predictions.max()}")
    logging.info(f"Mean: {final_predictions.mean()}")
    logging.info(f"Median: {np.median(final_predictions)}")

if __name__ == '__main__':
    main()