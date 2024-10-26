import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import logging
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SGCarPricePredictor:
    def __init__(self, n_folds=5, n_bags=3, seed=42):
        self.n_folds = n_folds
        self.n_bags = n_bags
        self.seed = seed
        self.scaler = None  # 可能不需要，但保留以备不时之需

        # 定义特征
        self.bool_cols = [
            'vehicle_type_bus_mini_bus', 'vehicle_type_hatchback', 
            'vehicle_type_luxury_sedan', 'vehicle_type_midsized_sedan',
            'vehicle_type_mpv', 'vehicle_type_others', 'vehicle_type_sports_car',
            'vehicle_type_stationwagon', 'vehicle_type_suv', 'vehicle_type_truck',
            'vehicle_type_van', 'transmission_type_auto', 'transmission_type_manual',
            'almost new car', 'coe car', 'consignment car', 'direct owner sale',
            'electric cars', 'hybrid cars', 'imported used vehicle', 'low mileage car',
            'opc car', 'parf car', 'premium ad car', 'rare & exotic',
            'sgcarmart warranty cars', 'sta evaluated car', 'vintage cars'
        ]
        self.num_cols = [
            'manufactured', 'curb_weight', 'power', 'engine_cap', 'no_of_owners',
            'depreciation', 'coe', 'road_tax', 'dereg_value', 'omv', 'arf', 'vehicle_age'
        ]

    def clean_data(self, df):
        """数据清洗：移除数值特征中的异常值"""
        data = df.copy()
        for col in self.num_cols:
            if col in data.columns:
                q1 = data[col].quantile(0.01)
                q3 = data[col].quantile(0.99)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
        return data

    def create_price_clusters(self, df, n_clusters=5):
        """创建价格聚类以辅助模型"""
        key_features = ['depreciation', 'coe', 'dereg_value', 'omv', 'arf']
        scaler = StandardScaler()
        cluster_features = scaler.fit_transform(df[key_features])
        log_price = np.log1p(df['price'])
        cluster_data = np.column_stack([cluster_features, log_price])

        kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=10)
        clusters = kmeans.fit_predict(cluster_data)

        # 打印聚类统计信息
        cluster_stats = df.groupby(clusters).agg({
            'price': ['count', 'mean'],
            'depreciation': 'mean',
            'coe': 'mean'
        }).reset_index()
        cluster_stats.columns = ['cluster', 'count', 'avg_price', 'avg_depreciation', 'avg_coe']
        logging.info("\nCluster Statistics:")
        logging.info(cluster_stats)

        return clusters

    def preprocess_features(self, df, is_train=True):
        """特征预处理：移除不必要的特征工程步骤，保留关键特征"""
        data = df.copy()

        # 计算车辆制造年份
        current_year = datetime.now().year
        data['manufactured_year'] = current_year - data['vehicle_age']

        # 创建车辆总成本和价值保留率
        data['total_cost'] = data['omv'] + data['arf'] + data['coe']
        data['value_retention'] = data['dereg_value'] / np.maximum(data['total_cost'], 1)

        # 创建比率特征
        data['power_weight_ratio'] = data['power'] / np.maximum(data['curb_weight'], 1)
        data['depreciation_ratio'] = data['depreciation'] / np.maximum(data['omv'], 1)
        data['coe_omv_ratio'] = data['coe'] / np.maximum(data['omv'], 1)
        data['dereg_value_ratio'] = data['dereg_value'] / np.maximum(data['total_cost'], 1)
        data['tax_engine_ratio'] = data['road_tax'] / np.maximum(data['engine_cap'], 1)

        # 创建组合特征
        data['is_luxury'] = ((data['vehicle_type_luxury_sedan'] == 1) | 
                              (data['rare & exotic'] == 1)).astype(int)
        data['is_economic'] = ((data['vehicle_type_hatchback'] == 1) | 
                                (data['vehicle_type_midsized_sedan'] == 1)).astype(int)
        data['luxury_power'] = data['is_luxury'] * data['power']
        data['eco_efficiency'] = data['is_economic'] * data['engine_cap']
        data['age_power'] = data['vehicle_age'] * data['power']

        # 创建价格相关特征（仅用于训练集）
        if is_train and 'price' in data.columns:
            data['log_price'] = np.log1p(data['price'])
            try:
                data['price_bin'] = pd.qcut(data['price'], q=10, labels=False, duplicates='drop')
            except Exception as e:
                logging.warning(f"Error in creating price bins: {str(e)}")
                data['price_bin'] = 0

        return data

    def train_model(self, model_params, train, test, target_col, model_type='lightgbm'):
        """模型训练：支持LightGBM和CatBoost两种模型类型"""
        oof_preds = np.zeros(len(train))
        test_preds = np.zeros(len(test))
        feature_importance = pd.DataFrame()

        # 定义特征列表
        base_features = self.bool_cols + self.num_cols + [
            'depreciation_ratio', 'coe_omv_ratio', 'dereg_value_ratio', 
            'tax_engine_ratio', 'total_cost', 'value_retention', 
            'manufactured_year', 'is_luxury', 'is_economic', 
            'luxury_power', 'eco_efficiency', 'age_power'
        ]

        # 确保所有基础特征都在数据中存在
        features = [feat for feat in base_features if feat in train.columns]

        # 交叉验证
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
            start_time = time.time()

            X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
            y_train = np.log1p(X_train[target_col])
            y_val = np.log1p(X_val[target_col])

            fold_test_preds = np.zeros((len(test), self.n_bags))
            fold_val_preds = np.zeros((len(val_idx), self.n_bags))

            for bag in range(self.n_bags):
                bag_seed = self.seed + bag + fold * 100

                if model_type == 'lightgbm':
                    model = lgb.LGBMRegressor(**model_params, random_state=bag_seed)
                    model.fit(
                        X_train[features], y_train,
                        eval_set=[(X_val[features], y_val)],
                        # early_stopping_rounds=50,
                        # verbose=False
                    )
                elif model_type == 'catboost':
                    model = CatBoostRegressor(**model_params, random_seed=bag_seed)
                    model.fit(
                        X_train[features], y_train,
                        eval_set=(X_val[features], y_val),
                        early_stopping_rounds=50
                    )
                else:
                    raise ValueError("Unsupported model type. Choose 'lightgbm' or 'catboost'.")

                # 预测并转换回原始尺度
                fold_val_preds[:, bag] = np.expm1(model.predict(X_val[features]))
                fold_test_preds[:, bag] = np.expm1(model.predict(test[features]))

                # 收集特征重要性
                if hasattr(model, 'feature_importances_'):
                    importance = pd.DataFrame({
                        'feature': features,
                        'importance': model.feature_importances_
                    })
                    feature_importance = pd.concat([feature_importance, importance], axis=0)

            # 平均袋装预测
            oof_preds[val_idx] = fold_val_preds.mean(axis=1)
            test_preds += fold_test_preds.mean(axis=1) / self.n_folds

            # 计算当前折的分数
            fold_score = np.sqrt(mean_squared_error(
                X_val[target_col], 
                oof_preds[val_idx]
            ))
            elapsed = time.time() - start_time
            logging.info(f'Fold {fold+1} - RMSE: {fold_score:.4f} - Time: {elapsed:.2f}s')

        # 打印特征重要性
        if not feature_importance.empty:
            mean_importance = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
            logging.info("\nTop 10 important features:")
            logging.info(mean_importance.head(10))

        return oof_preds, test_preds

    def fit_predict(self, train_df, test_df):
        """主训练和预测流程"""
        start_time = time.time()

        # 数据清洗
        train_df = self.clean_data(train_df)
        test_df = self.clean_data(test_df)

        # 预处理
        train = self.preprocess_features(train_df, is_train=True)
        test = self.preprocess_features(test_df, is_train=False)

        # 创建聚类（可选，根据实际效果决定是否使用）
        # clusters = self.create_price_clusters(train)
        # train['cluster'] = clusters

        # 定义模型参数，移除 'random_state' 和 'random_seed'
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'n_estimators': 2000,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 6,
            'min_child_samples': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'min_gain_to_split': 0.1,
            'min_sum_hessian_in_leaf': 1.0,
            # 'random_state': self.seed,  # 移除此行
            'verbosity': -1
        }

        cat_params = {
            'iterations': 2000,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 5,
            'min_data_in_leaf': 20,
            'random_strength': 0.5,
            'bagging_temperature': 0.2,
            'od_type': 'Iter',
            'od_wait': 50,
            # 'random_seed': self.seed,  # 移除此行
            'verbose': False
        }

        # 训练 LightGBM 模型
        logging.info("\nTraining LightGBM model...")
        lgb_oof, lgb_test = self.train_model(lgb_params, train, test, 'price', model_type='lightgbm')

        # 训练 CatBoost 模型
        logging.info("\nTraining CatBoost model...")
        cat_oof, cat_test = self.train_model(cat_params, train, test, 'price', model_type='catboost')

        # 融合预测
        final_oof = 0.7 * lgb_oof + 0.3 * cat_oof
        final_test = 0.7 * lgb_test + 0.3 * cat_test

        # 后处理预测结果，确保在合理范围内
        final_test = np.clip(final_test, train_df['price'].min(), train_df['price'].max())

        elapsed_time = time.time() - start_time
        logging.info(f'\nTotal training time: {elapsed_time/60:.2f} minutes')

        # 计算最终验证分数
        final_score = np.sqrt(mean_squared_error(np.log1p(train_df['price']), np.log1p(final_oof)))
        logging.info(f'Final validation RMSE (log scale): {final_score:.4f}')

        return final_oof, final_test

if __name__ == "__main__":
    # 加载数据
    train_df = pd.read_csv("preprocessing/2024-10-21-silan/train_cleaned.csv")
    test_df = pd.read_csv("preprocessing/2024-10-21-silan/test_cleaned.csv")

    # 初始化和运行模型
    predictor = SGCarPricePredictor(n_folds=5, n_bags=3)
    oof_predictions, test_predictions = predictor.fit_predict(train_df, test_df)

    # 创建提交文件
    submission = pd.DataFrame({
        'Id': test_df['Id'] if 'Id' in test_df.columns else range(len(test_predictions)),
        'Predicted': np.round(test_predictions).astype(int)
    })
    submission.to_csv('1025_submission_mixed.csv', index=False)