import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import logging
import time
import pickle
from category_encoders import TargetEncoder

# 设置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
            self.weights = self.weights / np.sum(self.weights)
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

def clean_feature_names(X):
    """清理特征名称，移除或替换特殊字符"""
    X = X.copy()
    feature_map = {}
    for col in X.columns:
        new_name = (col.replace('[', '_')
                     .replace(']', '_')
                     .replace('<', '_')
                     .replace('>', '_')
                     .replace(':', '_')
                     .replace('"', '_')
                     .replace('/', '_')
                     .replace('\\', '_')
                     .replace('|', '_')
                     .replace('?', '_')
                     .replace('*', '_')
                     .replace(' ', '_'))
        
        if new_name in feature_map.values():
            i = 1
            while f"{new_name}_{i}" in feature_map.values():
                i += 1
            new_name = f"{new_name}_{i}"
        
        feature_map[col] = new_name
    
    X.columns = [feature_map[col] for col in X.columns]
    return X, feature_map

class OptimizedPricePredictor:
    def __init__(self, price_threshold_percentile=75, n_splits=5, random_state=42):
        self.price_threshold_percentile = price_threshold_percentile
        self.n_splits = n_splits
        self.random_state = random_state
        self.num_imputer = None
        self.cat_imputer = None
        self.scaler = None
        self.target_encoder = None
        self.feature_map = None
        self.price_threshold = None
        self.models = None
        self.ensemble_weights = None
    
    def preprocess_features(self, X, y=None, fit=False):
        """特征预处理"""
        X = X.copy()
        
        # 清理特征名称
        if fit:
            X, self.feature_map = clean_feature_names(X)
        else:
            if self.feature_map:
                X.columns = [self.feature_map.get(col, col) for col in X.columns]
        
        # 分离数值和分类特征
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        # 处理数值特征
        if fit:
            self.num_imputer = SimpleImputer(strategy='median')
            self.scaler = StandardScaler()
        
        if len(numeric_features) > 0:
            X[numeric_features] = self.num_imputer.fit_transform(X[numeric_features]) if fit else \
                                self.num_imputer.transform(X[numeric_features])
            
            # 标准化特定特征
            scale_features = ['curb_weight', 'power', 'engine_cap', 'depreciation']
            scale_features = [self.feature_map.get(f, f) for f in scale_features if self.feature_map.get(f, f) in X.columns]
            if scale_features:
                X[scale_features] = self.scaler.fit_transform(X[scale_features]) if fit else \
                                  self.scaler.transform(X[scale_features])
        
        # 处理分类特征
        if len(categorical_features) > 0:
            if fit:
                self.cat_imputer = SimpleImputer(strategy='constant', fill_value='unknown')
                self.target_encoder = TargetEncoder(smoothing=10)
            
            X[categorical_features] = self.cat_imputer.fit_transform(X[categorical_features]) if fit else \
                                    self.cat_imputer.transform(X[categorical_features])
            
            if y is not None:
                X[categorical_features] = self.target_encoder.fit_transform(X[categorical_features], y) if fit else \
                                        self.target_encoder.transform(X[categorical_features])
        
        return X
    
    def create_essential_features(self, X):
        """创建核心特征交互"""
        X = X.copy()
        
        ref_price = self.feature_map.get('ref_price', 'ref_price')
        depreciation = self.feature_map.get('depreciation', 'depreciation')
        dereg_value = self.feature_map.get('dereg_value', 'dereg_value')
        coe = self.feature_map.get('coe', 'coe')
        
        # 保留最重要的特征交互
        if ref_price in X.columns and depreciation in X.columns:
            X['depreciation_rate'] = X[depreciation] / (X[ref_price] + 1)
            
        if coe in X.columns and dereg_value in X.columns:
            X['coe_dereg_ratio'] = X[coe] / (X[dereg_value] + 1)
        
        return X
    
    def create_price_clusters(self, y):
        """基于价格阈值的双聚类"""
        if self.price_threshold is None:
            self.price_threshold = np.percentile(y, self.price_threshold_percentile)
        
        clusters = (y > self.price_threshold).astype(int)
        
        # 输出聚类信息
        cluster_info = []
        for cluster in range(2):
            cluster_prices = y[clusters == cluster]
            cluster_info.append({
                'cluster': cluster,
                'min': cluster_prices.min(),
                'max': cluster_prices.max(),
                'median': cluster_prices.median(),
                'count': len(cluster_prices),
                'mean': cluster_prices.mean(),
                'std': cluster_prices.std()
            })
        
        cluster_df = pd.DataFrame(cluster_info)
        logging.info("Price Cluster Information:")
        logging.info(cluster_df)
        
        return clusters

    def train_lightgbm(self, X_train, y_train, X_val, y_val, cluster):
        """训练LightGBM模型"""
        train_data = lgb.Dataset(X_train, label=np.log1p(y_train))
        val_data = lgb.Dataset(X_val, label=np.log1p(y_val), reference=train_data)
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'seed': self.random_state,
        }
        
        if cluster == 0:  # 低价格群体
            params.update({
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.7,
                'bagging_freq': 5,
                'min_child_samples': 20,
            })
        else:  # 高价格群体
            params.update({
                'num_leaves': 21,
                'learning_rate': 0.03,
                'feature_fraction': 0.7,
                'bagging_fraction': 0.8,
                'bagging_freq': 4,
                'min_child_samples': 30,
            })
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
        )
        
        return model

    def train_xgboost(self, X_train, y_train, X_val, y_val, cluster):
        """训练XGBoost模型"""
        dtrain = xgb.DMatrix(X_train, label=np.log1p(y_train))
        dval = xgb.DMatrix(X_val, label=np.log1p(y_val))
        
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'seed': self.random_state,
        }
        
        if cluster == 0:  # 低价格群体
            params.update({
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
            })
        else:  # 高价格群体
            params.update({
                'max_depth': 5,
                'learning_rate': 0.03,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'min_child_weight': 3,
            })
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, 'val')],
        )
        
        return model

    def train_catboost(self, X_train, y_train, X_val, y_val, cluster):
        """训练CatBoost模型"""
        params = {
            'iterations': 1000,
            'random_seed': self.random_state,
            'od_type': 'Iter',
            'od_wait': 50,
            'verbose': False
        }
        
        if cluster == 0:  # 低价格群体
            params.update({
                'depth': 6,
                'learning_rate': 0.05,
                'l2_leaf_reg': 3,
                'subsample': 0.8,
            })
        else:  # 高价格群体
            params.update({
                'depth': 5,
                'learning_rate': 0.03,
                'l2_leaf_reg': 5,
                'subsample': 0.7,
            })
        
        model = CatBoostRegressor(**params)
        model.fit(
            X_train, 
            np.log1p(y_train),
            eval_set=(X_val, np.log1p(y_val)),
            verbose=False
        )
        
        return model

    def train_gradientboost(self, X_train, y_train, X_val, y_val, cluster):
        """训练GradientBoosting模型"""
        params = {
            'random_state': self.random_state,
            'validation_fraction': 0.1,
            'n_iter_no_change': 50,
            'tol': 1e-4
        }
        
        if cluster == 0:  # 低价格群体
            params.update({
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'max_depth': 5,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'subsample': 0.8,
                'loss': 'huber',
                'alpha': 0.9
            })
        else:  # 高价格群体
            params.update({
                'n_estimators': 800,
                'learning_rate': 0.03,
                'max_depth': 4,
                'min_samples_split': 30,
                'min_samples_leaf': 15,
                'subsample': 0.7,
                'loss': 'huber',
                'alpha': 0.95
            })
        
        model = GradientBoostingRegressor(**params)
        model.fit(X_train, np.log1p(y_train))
        
        return model

    def fit(self, X, y):
        """训练完整模型"""
        start_time = time.time()
        
        # 预处理特征
        X_processed = self.preprocess_features(X, y, fit=True)
        X_processed = self.create_essential_features(X_processed)
        
        # 创建价格聚类
        clusters = self.create_price_clusters(y)
        
        # 初始化K-Fold
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        # 存储模型和验证结果
        self.models = {0: [], 1: []}  # 分别存储低价格和高价格模型
        oof_predictions = np.zeros(len(X))
        feature_importances = []
        
        # 存储每个聚类的WeightedEnsembleL2
        self.ensemble_weights = {0: [], 1: []}
        
        # 对每个聚类训练模型
        for cluster in range(2):
            cluster_mask = clusters == cluster
            X_cluster = X_processed[cluster_mask]
            y_cluster = y[cluster_mask]
            
            if len(X_cluster) < self.n_splits:
                logging.warning(f"Cluster {cluster} has too few samples ({len(X_cluster)}). Skipping...")
                continue
            
            logging.info(f"\nTraining models for Cluster {cluster}")
            
            # K-Fold交叉验证
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_cluster), 1):
                X_train, X_val = X_cluster.iloc[train_idx], X_cluster.iloc[val_idx]
                y_train, y_val = y_cluster.iloc[train_idx], y_cluster.iloc[val_idx]
                
                # 训练所有模型
                lgb_model = self.train_lightgbm(X_train, y_train, X_val, y_val, cluster)
                xgb_model = self.train_xgboost(X_train, y_train, X_val, y_val, cluster)
                cb_model = self.train_catboost(X_train, y_train, X_val, y_val, cluster)
                gb_model = self.train_gradientboost(X_train, y_train, X_val, y_val, cluster)
                
                # 存储模型
                self.models[cluster].append({
                    'lightgbm': lgb_model,
                    'xgboost': xgb_model,
                    'catboost': cb_model,
                    'gradientboost': gb_model,
                    'features': X_train.columns.tolist()
                })
                
                # 验证集预测
                val_preds_lgb = np.expm1(lgb_model.predict(X_val))
                val_preds_xgb = np.expm1(xgb_model.predict(xgb.DMatrix(X_val)))
                val_preds_cb = np.expm1(cb_model.predict(X_val))
                val_preds_gb = np.expm1(gb_model.predict(X_val))
                
                # 使用WeightedEnsembleL2计算权重
                val_predictions = np.column_stack([val_preds_lgb, val_preds_xgb, val_preds_cb, val_preds_gb])
                ensemble = WeightedEnsembleL2()
                ensemble.fit(val_predictions, y_val)
                
                # 存储ensemble权重
                self.ensemble_weights[cluster].append(ensemble.weights)
                
                # 加权预测
                val_preds = ensemble.predict(val_predictions)
                
                # 应用价格约束
                if cluster == 0:  # 低价格群体
                    val_preds = np.clip(val_preds, 700, self.price_threshold)
                else:  # 高价格群体
                    val_preds = np.clip(val_preds, self.price_threshold, 2900000)
                
                # 存储OOF预测
                oof_predictions[cluster_mask][val_idx] = val_preds
                
                # 记录性能
                rmse = np.sqrt(mean_squared_error(y_val, val_preds))
                r2 = r2_score(y_val, val_preds)
                logging.info(f"Cluster {cluster} - Fold {fold} - RMSE: {rmse:.2f}, R2: {r2:.4f}")
                logging.info(f"Ensemble weights: LGB={ensemble.weights[0]:.3f}, XGB={ensemble.weights[1]:.3f}, "
                           f"CB={ensemble.weights[2]:.3f}, GB={ensemble.weights[3]:.3f}")
                
                # 记录特征重要性
                importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': gb_model.feature_importances_
                })
                feature_importances.append(importance)
        
        # 计算并输出总体性能
        overall_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
        overall_r2 = r2_score(y, oof_predictions)
        logging.info(f"\nOverall RMSE: {overall_rmse:.2f}")
        logging.info(f"Overall R2: {overall_r2:.4f}")
        
        # 汇总特征重要性
        self.feature_importance = pd.concat(feature_importances).groupby('feature').mean()
        self.feature_importance = self.feature_importance.sort_values('importance', ascending=False)
        
        logging.info(f"\nTop 10 important features:")
        logging.info(self.feature_importance.head(10))
        
        training_time = (time.time() - start_time) / 60
        logging.info(f"\nTotal training time: {training_time:.2f} minutes")
        
        return self

    def predict(self, X):
        """预测新数据"""
        # 预处理特征
        X_processed = self.preprocess_features(X, fit=False)
        X_processed = self.create_essential_features(X_processed)
        
        # 初始化预测结果
        predictions = np.zeros(len(X))
        prediction_counts = np.zeros(len(X))
        
        # 基于ref_price预测聚类概率
        ref_price_col = self.feature_map.get('ref_price', 'ref_price')
        if ref_price_col in X_processed.columns:
            ref_prices = X_processed[ref_price_col]
            prob_high = 1 / (1 + np.exp(-0.1 * (ref_prices - np.mean(ref_prices))))
        else:
            prob_high = np.full(len(X), 0.5)
        
        # 对每个聚类进行预测
        for cluster in [0, 1]:
            cluster_predictions = []
            
            if cluster in self.models:
                for model_dict, ensemble_weight in zip(self.models[cluster], self.ensemble_weights[cluster]):
                    # 确保特征顺序一致
                    features = model_dict['features']
                    missing_features = set(features) - set(X_processed.columns)
                    if missing_features:
                        for feature in missing_features:
                            X_processed[feature] = 0
                    
                    X_model = X_processed[features]
                    
                    # 各模型预测
                    lgb_pred = np.expm1(model_dict['lightgbm'].predict(X_model))
                    xgb_pred = np.expm1(model_dict['xgboost'].predict(xgb.DMatrix(X_model)))
                    cb_pred = np.expm1(model_dict['catboost'].predict(X_model))
                    gb_pred = np.expm1(model_dict['gradientboost'].predict(X_model))
                    
                    # 使用保存的ensemble权重进行预测
                    fold_predictions = np.column_stack([lgb_pred, xgb_pred, cb_pred, gb_pred])
                    fold_preds = fold_predictions @ ensemble_weight
                    
                    # 应用价格约束
                    if cluster == 0:
                        fold_preds = np.clip(fold_preds, 700, self.price_threshold)
                    else:
                        fold_preds = np.clip(fold_preds, self.price_threshold, 2900000)
                    
                    cluster_predictions.append(fold_preds)
            
            if cluster_predictions:
                # 计算该聚类的平均预测
                cluster_preds = np.mean(cluster_predictions, axis=0)
                
                # 根据cluster调整预测权重
                if cluster == 0:  # 低价格群体
                    weights = 1 - prob_high
                else:  # 高价格群体
                    weights = prob_high
                
                predictions += weights * cluster_preds
                prediction_counts += weights
        
        # 处理权重和为0的情况
        mask_zero_counts = prediction_counts == 0
        if np.any(mask_zero_counts):
            predictions[mask_zero_counts] = np.mean(predictions[~mask_zero_counts])
        else:
            predictions = predictions / prediction_counts
        
        # 最终的预测约束
        predictions = np.clip(predictions, 700, 2900000)
        
        return predictions

def main():
    """主函数"""
    # 设置随机种子
    np.random.seed(42)
    
    # 加载数据
    logging.info("Loading data...")
    try:
        train_data = pd.read_csv('preprocessing/2024-10-30-silan/train_cleaned.csv')
        test_data = pd.read_csv('preprocessing/2024-10-30-silan/test_cleaned.csv')
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        return
    
    # 准备训练数据
    X_train = train_data.drop('price', axis=1)
    y_train = train_data['price']
    X_test = test_data.copy()
    
    # 输出数据信息
    logging.info(f"Training data shape: {X_train.shape}")
    logging.info(f"Test data shape: {X_test.shape}")
    logging.info("\nTarget variable statistics:")
    logging.info(y_train.describe())
    
    # 初始化并训练模型
    model = OptimizedPricePredictor(
        price_threshold_percentile=75,  # 使用75分位数作为价格阈值
        n_splits=5,
        random_state=42
    )
    
    try:
        logging.info("\nTraining model...")
        model.fit(X_train, y_train)
        
        # 保存模型
        logging.info("\nSaving model...")
        with open('optimized_price_predictor.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        # 预测测试集
        logging.info("\nPredicting test data...")
        test_predictions = model.predict(X_test)
        
        # 创建提交文件
        submission = pd.DataFrame({
            'Id': range(len(test_predictions)),
            'Predicted': np.round(test_predictions).astype(int)
        })
        
        submission.to_csv('submission_optimized_model.csv', index=False)
        
        # 输出预测统计信息
        logging.info("\nPrediction statistics:")
        logging.info(f"Minimum: {test_predictions.min():.2f}")
        logging.info(f"Maximum: {test_predictions.max():.2f}")
        logging.info(f"Mean: {test_predictions.mean():.2f}")
        logging.info(f"Median: {np.median(test_predictions):.2f}")
        
    except Exception as e:
        logging.error(f"Error during model training or prediction: {str(e)}")
        raise

if __name__ == '__main__':
    main()