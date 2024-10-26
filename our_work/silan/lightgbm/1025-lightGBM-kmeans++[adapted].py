import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import logging
import time
from datetime import datetime
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SGCarPricePredictor:
    def __init__(self, n_folds=5, n_bags=3, seed=42):
        self.n_folds = n_folds
        self.n_bags = n_bags
        self.seed = seed
        self.scaler = None
        
        # 特征定义
        self.bool_cols = [
            'vehicle_type_bus_mini_bus', 'vehicle_type_hatchback', 
            'vehicle_type_luxury_sedan', 'vehicle_type_midsized_sedan',
            'vehicle_type_mpv', 'vehicle_type_others', 'vehicle_type_sports_car',
            'vehicle_type_stationwagon', 'vehicle_type_suv', 'vehicle_type_truck',
            'vehicle_type_van', 'transmission_type_auto', 'transmission_type_manual',
            'almost_new_car', 'coe_car', 'consignment_car', 'direct_owner_sale',
            'electric_cars', 'hybrid_cars', 'imported_used_vehicle', 'low_mileage_car',
            'opc_car', 'parf_car', 'premium_ad_car', 'rare_and_exotic',
            'sgcarmart_warranty_cars', 'sta_evaluated_car', 'vintage_cars'
        ]
        self.num_cols = [
            'manufactured', 'curb_weight', 'power', 'engine_cap', 'no_of_owners',
            'depreciation', 'coe', 'road_tax', 'dereg_value', 'omv', 'arf', 'vehicle_age'
        ]
    
    @staticmethod
    def clean_feature_name(name):
        """清理特征名称，确保与LightGBM兼容"""
        # 将特殊字符替换为下划线
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(name))
        # 确保名称以字母开头
        if clean_name[0].isdigit():
            clean_name = 'n_' + clean_name
        # 移除连续的下划线
        clean_name = re.sub(r'_+', '_', clean_name)
        # 移除首尾的下划线
        clean_name = clean_name.strip('_')
        return clean_name.lower()

    def clean_data(self, df):
        """增强的数据清洗"""
        data = df.copy()
        
        # 清理所有列名
        data.columns = [self.clean_feature_name(col) for col in data.columns]
        
        # 1. 异常值处理（使用IQR方法）
        num_cols = [self.clean_feature_name(col) for col in self.num_cols]
        for col in num_cols:
            if col in data.columns:
                q1 = data[col].quantile(0.01)
                q3 = data[col].quantile(0.99)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
        
        # 2. 添加价格分层
        if 'price' in data.columns:
            data['price_category'] = pd.qcut(data['price'], q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
            
        return data

    def train_model(self, model_params, train, test, target_col, cluster_weights=None):
        """使用清理过的特征名称的模型训练"""
        oof_preds = np.zeros(len(train))
        test_preds = np.zeros(len(test))
        feature_importance = pd.DataFrame()
        
        # 排除特定列
        exclude_cols = ['price', 'logprice', 'pricebin', 'price_category', 'cluster']
        numeric_features = train.select_dtypes(include=['int64', 'float64', 'bool']).columns
        features = [col for col in numeric_features if col not in exclude_cols]
        
        # 确保所有特征名称都经过清理
        features = [self.clean_feature_name(f) for f in features]
        
        # 确保布尔型转为整型
        for col in features:
            if col in train.columns and train[col].dtype == bool:
                train[col] = train[col].astype(int)
            if col in test.columns and test[col].dtype == bool:
                test[col] = test[col].astype(int)
        
        # 确保所有特征都存在于两个数据集中
        common_features = [f for f in features if f in train.columns and f in test.columns]
        
        # 填充缺失值
        train[common_features] = train[common_features].fillna(-999)
        test[common_features] = test[common_features].fillna(-999)
        
        # 交叉验证
        kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(train, train['pricebin'])):
            start_time = time.time()
            
            X_train = train.iloc[train_idx]
            X_val = train.iloc[val_idx]
            y_train = np.log1p(X_train[target_col])
            y_val = np.log1p(X_val[target_col])
            
            if cluster_weights is not None:
                sample_weights = np.ones(len(X_train))
                for cluster, weight in enumerate(cluster_weights):
                    sample_weights[X_train['cluster'] == cluster] = weight
            else:
                sample_weights = None
            
            fold_test_preds = np.zeros((len(test), self.n_bags))
            fold_val_preds = np.zeros((len(val_idx), self.n_bags))
            
            for bag in range(self.n_bags):
                bag_seed = self.seed + bag + fold * 100
                
                if 'boosting_type' in model_params:
                    # LightGBM
                    model = lgb.LGBMRegressor(**model_params, random_state=bag_seed)
                    model.fit(
                        X_train[common_features], y_train,
                        eval_set=[(X_val[common_features], y_val)],
                        sample_weight=sample_weights,
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=0)]
                    )
                else:
                    # CatBoost
                    model_params_copy = model_params.copy()
                    model_params_copy['verbose'] = 0
                    model = CatBoostRegressor(**model_params_copy, random_seed=bag_seed)
                    model.fit(
                        X_train[common_features], y_train,
                        eval_set=[(X_val[common_features], y_val)],
                        sample_weight=sample_weights,
                        early_stopping_rounds=50,
                        verbose=False
                    )
                
                fold_val_preds[:, bag] = np.expm1(model.predict(X_val[common_features]))
                fold_test_preds[:, bag] = np.expm1(model.predict(test[common_features]))
                
                if hasattr(model, 'feature_importances_'):
                    importance = pd.DataFrame({
                        'feature': common_features,
                        'importance': model.feature_importances_
                    })
                    feature_importance = pd.concat([feature_importance, importance])
            
            oof_preds[val_idx] = fold_val_preds.mean(axis=1)
            test_preds += fold_test_preds.mean(axis=1) / self.n_folds
            
            fold_score = np.sqrt(mean_squared_error(X_val[target_col], oof_preds[val_idx]))
            elapsed = time.time() - start_time
            logging.info(f'Fold {fold+1} - RMSE: {fold_score:.0f} - Time: {elapsed:.0f}s')
        
        if not feature_importance.empty:
            mean_importance = feature_importance.groupby('feature')['importance'].mean()
            logging.info("\nTop 10 important features:")
            logging.info(mean_importance.sort_values(ascending=False).head(10))
        
        return oof_preds, test_preds
    
    def preprocess_features(self, df, is_train=True):
        """修改后的特征工程"""
        data = df.copy()
        
        
        # 2. 标准化数值特征 - 修改需要标准化的特征列表
        scale_features = ['curb_weight', 'power', 'engine_cap', 'depreciation']
        
        if is_train:
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(data[scale_features])
        else:
            if self.scaler is None:
                raise ValueError("Scaler is not fitted! Run with is_train=True first.")
            scaled_features = self.scaler.transform(data[scale_features])
        
        for i, col in enumerate(scale_features):
            data[f'scaled{col}'] = scaled_features[:, i]
        
        # 3. 创建新的交互特征
        data['totalcost'] = data['omv'] + data['arf'] + data['coe']
        data['valueretention'] = data['dereg_value'] / np.maximum(data['totalcost'], 1)
        data['powerweight'] = data['power'] / np.maximum(data['curb_weight'], 1)
        data['priceperyear'] = data['depreciation'] * data['vehicle_age']
        data['totaltax'] = data['road_tax'] + data['arf']
        
        # 4. 车龄相关特征
        data['agesquared'] = data['vehicle_age'] ** 2
        data['manufacturedyear'] = datetime.now().year - data['vehicle_age']
        data['agedep'] = data['vehicle_age'] * data['depreciation']
        
        # 5. 市场定位特征
        data['marketpos'] = data['omv'] / np.maximum(data['engine_cap'], 1)
        data['luxindex'] = (data['arf'] + data['coe']) / np.maximum(data['omv'], 1)
        
        # 6. 品牌溢价指标
        data['premiumratio'] = data['omv'] / (data['engine_cap'] * data['power'])
        
        # 7. 车辆保值指标
        data['valuepreserve'] = data['dereg_value'] / (data['depreciation'] * data['vehicle_age'])
        
        if is_train and 'price' in data.columns:
            data['logprice'] = np.log1p(data['price'])
            data['pricebin'] = pd.qcut(data['price'], q=10, labels=False, duplicates='drop')
            
        return data

    def create_price_clusters(self, df, n_clusters=5):
        """改进的聚类方法"""
        key_features = ['depreciation', 'coe', 'dereg_value', 'omv', 'arf']
        
        scaler = StandardScaler()
        cluster_features = scaler.fit_transform(df[key_features])
        
        # 对价格取对数以减少极端值影响
        log_price = np.log1p(df['price'])
        cluster_data = np.column_stack([cluster_features, log_price])
        
        # 使用KMeans聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=10)
        clusters = kmeans.fit_predict(cluster_data)
        
        # 计算每个簇的统计信息
        cluster_stats = []
        for cluster in range(n_clusters):
            mask = clusters == cluster
            cluster_stats.append({
                'cluster': cluster,
                'count': mask.sum(),
                'avg_price': df.loc[mask, 'price'].mean(),
                'avg_depreciation': df.loc[mask, 'depreciation'].mean(),
                'avg_coe': df.loc[mask, 'coe'].mean()
            })
        
        # 计算簇间价格比例，用于后续加权
        total_records = df.shape[0]
        for stat in cluster_stats:
            stat['weight'] = np.sqrt(total_records / stat['count'])
        
        logging.info("\nCluster Statistics with Weights:")
        logging.info(pd.DataFrame(cluster_stats))
        
        return clusters, kmeans, cluster_stats

    def fit_predict(self, train_df, test_df):
        """改进的训练和预测流程"""
        start_time = time.time()
        
        # 1. 数据清洗
        train_df = self.clean_data(train_df)
        test_df = self.clean_data(test_df)
        
        # 2. 特征预处理
        train = self.preprocess_features(train_df, is_train=True)
        test = self.preprocess_features(test_df, is_train=False)
        
        # 3. 创建聚类并获取权重
        clusters, kmeans, cluster_stats = self.create_price_clusters(train)
        train['cluster'] = clusters
        cluster_weights = [stat['weight'] for stat in cluster_stats]
        
        # 4. 根据价格区间分层训练
        price_segments = pd.qcut(train['price'], q=3, labels=['low', 'medium', 'high'])
        models = {}
        predictions = {}
        
        for segment in ['low', 'medium', 'high']:
            logging.info(f"\nTraining models for {segment} price segment...")
            
            # 获取该价格段的训练数据
            segment_mask = price_segments == segment
            segment_train = train[segment_mask].copy()
            
            # 调整模型参数
            if segment == 'high':
                # 高价车型需要更保守的参数以避免过拟合
                lgb_params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'n_estimators': 2000,
                    'learning_rate': 0.02,
                    'num_leaves': 31,
                    'max_depth': 6,
                    'min_child_samples': 40,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'reg_alpha': 0.5,
                    'reg_lambda': 0.5,
                    'min_gain_to_split': 0.1,
                    'verbosity': -1
                }
                
                cat_params = {
                    'iterations': 2000,
                    'learning_rate': 0.02,
                    'depth': 6,
                    'l2_leaf_reg': 5,
                    'min_data_in_leaf': 40,
                    'random_strength': 0.5,
                    'bagging_temperature': 0.3,
                    'od_type': 'Iter',
                    'od_wait': 50,
                    'verbose': 0
                }
            else:
                # 中低价车型使用更激进的参数
                lgb_params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'n_estimators': 3000,
                    'learning_rate': 0.03,
                    'num_leaves': 40,
                    'max_depth': 8,
                    'min_child_samples': 30,
                    'feature_fraction': 0.7,
                    'bagging_fraction': 0.7,
                    'bagging_freq': 5,
                    'reg_alpha': 0.3,
                    'reg_lambda': 0.3,
                    'min_gain_to_split': 0.05,
                    'verbosity': -1
                }
                
                cat_params = {
                    'iterations': 3000,
                    'learning_rate': 0.03,
                    'depth': 8,
                    'l2_leaf_reg': 3,
                    'min_data_in_leaf': 30,
                    'random_strength': 0.3,
                    'bagging_temperature': 0.5,
                    'od_type': 'Iter',
                    'od_wait': 50,
                    'verbose': 0
                }
            
            # 训练模型
            logging.info(f"\nTraining LightGBM model for {segment} segment...")
            lgb_oof, lgb_test = self.train_model(
                lgb_params, 
                segment_train, 
                test, 
                'price',
                cluster_weights if segment == 'high' else None
            )
            
            logging.info(f"\nTraining CatBoost model for {segment} segment...")
            cat_oof, cat_test = self.train_model(
                cat_params, 
                segment_train, 
                test, 
                'price',
                cluster_weights if segment == 'high' else None
            )
            
            # 存储预测结果
            predictions[segment] = {
                'oof': 0.7 * lgb_oof + 0.3 * cat_oof,
                'test': 0.7 * lgb_test + 0.3 * cat_test
            }
        
        # 5. 合并预测结果
        final_oof = np.zeros(len(train))
        final_test = np.zeros(len(test))
        
        # 对于训练集，使用各自段的预测
        for segment in ['low', 'medium', 'high']:
            segment_mask = price_segments == segment
            final_oof[segment_mask] = predictions[segment]['oof']
        
        # 对于测试集，使用加权平均
        weights = {'low': 0.4, 'medium': 0.4, 'high': 0.2}
        for segment, weight in weights.items():
            final_test += predictions[segment]['test'] * weight
        
        # 6. 后处理预测结果
        price_bounds = {
            'low': (train_df['price'].quantile(0.01), train_df['price'].quantile(0.33)),
            'medium': (train_df['price'].quantile(0.33), train_df['price'].quantile(0.67)),
            'high': (train_df['price'].quantile(0.67), train_df['price'].quantile(0.99))
        }
        
        # 根据价格段限制预测范围
        final_test = np.clip(
            final_test,
            price_bounds['low'][0],
            price_bounds['high'][1]
        )
        
        elapsed_time = time.time() - start_time
        logging.info(f'\nTotal training time: {elapsed_time/60:.2f} minutes')
        
        # 计算并输出每个价格段的RMSE
        for segment in ['low', 'medium', 'high']:
            segment_mask = price_segments == segment
            segment_score = np.sqrt(mean_squared_error(
                train_df.loc[segment_mask, 'price'],
                final_oof[segment_mask]
            ))
            logging.info(f'{segment.capitalize()} price segment RMSE: {segment_score:.0f}')
        
        final_score = np.sqrt(mean_squared_error(train_df['price'], final_oof))
        logging.info(f'Final validation RMSE: {final_score:.0f}')
        
        return final_oof, final_test


if __name__ == "__main__":
    # Load data
    train_df = pd.read_csv("preprocessing/2024-10-21-silan/train_cleaned.csv")
    test_df = pd.read_csv("preprocessing/2024-10-21-silan/test_cleaned.csv")
    
    # Initialize and run model
    predictor = SGCarPricePredictor(n_folds=5, n_bags=3)
    oof_predictions, test_predictions = predictor.fit_predict(train_df, test_df)
    
    # Create submission
    submission = pd.DataFrame({
        'Id': range(len(test_predictions)),
        'Predicted': np.round(test_predictions).astype(int)
    })
    submission.to_csv('1025_submission_mixed_adapted.csv', index=False)