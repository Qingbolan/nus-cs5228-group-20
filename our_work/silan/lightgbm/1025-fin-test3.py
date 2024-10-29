import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
import logging
import time
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SGCarPricePredictor:
    def __init__(self, n_folds=5, n_bags=3, seed=42):
        self.n_folds = n_folds
        self.n_bags = n_bags
        self.seed = seed
        
        # 定义特征
        self.bool_cols = [
            'vehicle_type_bus_mini_bus', 'vehicle_type_hatchback', 
            'vehicle_type_luxury_sedan', 'vehicle_type_midsized_sedan',
            'vehicle_type_mpv', 'vehicle_type_others', 'vehicle_type_sports_car',
            'vehicle_type_stationwagon', 'vehicle_type_suv', 'vehicle_type_truck',
            'vehicle_type_van', 'transmission_type_auto', 'transmission_type_manual',
            'almost_new_car', 'coe_car', 'consignment_car', 'direct_owner_sale',
            'electric_cars', 'hybrid_cars', 'imported_used_vehicle', 'low_mileage_car',
            'opc_car', 'parf_car', 'premium_ad_car', 'rare_exotic',
            'sgcarmart_warranty_cars', 'sta_evaluated_car', 'vintage_cars'
        ]
        self.num_cols = [
            'manufactured', 'curb_weight', 'power', 'engine_cap', 'no_of_owners',
            'depreciation', 'road_tax', 'arf', 'vehicle_age'
        ]
        self.target_col = 'price'
        
        # 类别特征（如果有）
        self.categorical_cols = []  # 填写需要进行目标编码的类别特征

        # 预处理器
        self.preprocessor = None
        self._initialize_preprocessor()

    def _initialize_preprocessor(self):
        """初始化预处理流水线"""
        # 数值特征处理
        numeric_features = self.num_cols + [
            'manufactured_year', 'total_cost', 'value_retention', 
            'power_weight_ratio', 'depreciation_ratio', 'coe_omv_ratio', 
            'dereg_value_ratio', 'tax_engine_ratio', 'luxury_power', 
            'eco_efficiency', 'age_power'
        ]
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # 类别特征处理
        # 如果有需要目标编码的类别特征，添加到 categorical_transformer
        if self.categorical_cols:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
                ('target_encoder', TargetEncoder(cols=self.categorical_cols))
            ])
        else:
            categorical_transformer = 'passthrough'

        # 组合预处理器
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, self.categorical_cols)
            ],
            remainder='drop'
        )

    def clean_data(self, df):
        """数据清洗：移除数值特征中的异常值"""
        data = df.copy()
        for col in self.num_cols + ['coe', 'dereg_value', 'omv']:  # 特别注意可能的目标泄漏特征
            if col in data.columns:
                q1 = data[col].quantile(0.01)
                q3 = data[col].quantile(0.99)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
        return data

    def preprocess_features(self, df, is_train=True, target=None):
        """特征预处理，包括特征工程和预处理流水线"""
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
                              (data['rare_exotic'] == 1)).astype(int)
        data['is_economic'] = ((data['vehicle_type_hatchback'] == 1) | 
                                (data['vehicle_type_midsized_sedan'] == 1)).astype(int)
        data['luxury_power'] = data['is_luxury'] * data['power']
        data['eco_efficiency'] = data['is_economic'] * data['engine_cap']
        data['age_power'] = data['vehicle_age'] * data['power']

        # 移除可能引入目标信息的特征
        # 假设 'dereg_value' 和 'omv' 可能与 'price' 高度相关，建议移除或重新评估
        # 这里选择移除，以消除潜在的目标泄漏
        features_to_remove = ['dereg_value', 'omv']
        data.drop(columns=features_to_remove, inplace=True, errors='ignore')

        # 填补缺失值和标准化
        if is_train and target is not None:
            self.preprocessor.fit(data)
            data_preprocessed = self.preprocessor.transform(data)
        else:
            data_preprocessed = self.preprocessor.transform(data)

        # 构建最终的 DataFrame
        processed_features = self.preprocessor.get_feature_names_out()
        data_preprocessed = pd.DataFrame(data_preprocessed, columns=processed_features, index=df.index)

        return data_preprocessed

    def train_model(self, model_params, train, test, target_col, model_type='lightgbm'):
        """模型训练：支持 LightGBM、CatBoost 和 Sklearn 的 GradientBoostingRegressor 三种模型类型"""
        oof_preds = np.zeros(len(train))
        test_preds = np.zeros(len(test))
        feature_importance = pd.DataFrame()

        # 定义特征列表
        base_features = train.columns.tolist()

        # 交叉验证
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
            start_time = time.time()

            X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
            y_train = np.log1p(target_col.iloc[train_idx])
            y_val = np.log1p(target_col.iloc[val_idx])

            fold_test_preds = np.zeros((len(test), self.n_bags))
            fold_val_preds = np.zeros((len(val_idx), self.n_bags))

            for bag in range(self.n_bags):
                bag_seed = self.seed + bag + fold * 100

                if model_type == 'lightgbm':
                    model = lgb.LGBMRegressor(**model_params)
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                    )
                elif model_type == 'catboost':
                    model = CatBoostRegressor(**model_params)
                    model.fit(
                        X_train, y_train,
                        eval_set=(X_val, y_val),
                    )
                elif model_type == 'sklearn_gb':
                    model = GradientBoostingRegressor(**model_params)
                    model.fit(
                        X_train, y_train
                    )
                else:
                    raise ValueError("Unsupported model type. Choose 'lightgbm', 'catboost', or 'sklearn_gb'.")

                # 预测并转换回原始尺度
                if model_type in ['lightgbm', 'catboost']:
                    fold_val_preds[:, bag] = np.expm1(model.predict(X_val))
                    fold_test_preds[:, bag] = np.expm1(model.predict(test))
                elif model_type == 'sklearn_gb':
                    fold_val_preds[:, bag] = np.expm1(model.predict(X_val))
                    fold_test_preds[:, bag] = np.expm1(model.predict(test))

                # 收集特征重要性
                if hasattr(model, 'feature_importances_'):
                    importance = pd.DataFrame({
                        'feature': base_features,
                        'importance': model.feature_importances_,
                        'model': model_type,
                        'fold': fold + 1,
                        'bag': bag + 1
                    })
                    feature_importance = pd.concat([feature_importance, importance], axis=0)

            # 平均袋装预测
            oof_preds[val_idx] = fold_val_preds.mean(axis=1)
            test_preds += fold_test_preds.mean(axis=1) / self.n_folds

            # 计算当前折的分数
            fold_score = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
            elapsed = time.time() - start_time
            logging.info(f'Fold {fold+1} - Model: {model_type} - RMSE: {fold_score:.4f} - Time: {elapsed:.2f}s')

        # 打印特征重要性
        if not feature_importance.empty:
            mean_importance = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
            self.feature_importance = mean_importance
            logging.info("\nTop 10 important features:")
            logging.info(mean_importance.head(10))

        return oof_preds, test_preds

    def fit_predict(self, train_df, test_df):
        """主训练和预测流程"""
        start_time = time.time()

        # 数据清洗
        logging.info("Cleaning training data...")
        train_df = self.clean_data(train_df)
        logging.info("Cleaning test data...")
        test_df = self.clean_data(test_df)

        # 预处理
        logging.info("Preprocessing training data...")
        train = self.preprocess_features(train_df, is_train=True, target=train_df[self.target_col])
        logging.info("Preprocessing test data...")
        test = self.preprocess_features(test_df, is_train=False)

        # 特征列表更新（移除了与目标泄漏相关的特征）
        features = train.columns.tolist()

        # 定义模型参数
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'max_depth': -1,
            'min_child_samples': 20,
            'lambda_l2': 10,
            'random_state': self.seed,
        }

        cat_params = {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 10,
            'min_data_in_leaf': 20,
            'random_strength': 0.5,
            'bagging_temperature': 0.2,
            'od_type': 'Iter',
            'od_wait': 50,
            'random_seed': self.seed,
            'verbose': False
        }

        sklearn_gb_params = {
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'max_depth': 5,
            'min_samples_split': 20,
            'min_samples_leaf': 15,
            'loss': 'huber',
            'random_state': self.seed
        }

        # 训练 LightGBM 模型
        logging.info("\nTraining LightGBM model...")
        lgb_oof, lgb_test = self.train_model(lgb_params, train, test, train_df[self.target_col], model_type='lightgbm')

        # 训练 CatBoost 模型
        logging.info("\nTraining CatBoost model...")
        cat_oof, cat_test = self.train_model(cat_params, train, test, train_df[self.target_col], model_type='catboost')

        # 训练 Sklearn GradientBoostingRegressor 模型
        logging.info("\nTraining Sklearn GradientBoostingRegressor model...")
        sklearn_gb_oof, sklearn_gb_test = self.train_model(sklearn_gb_params, train, test, train_df[self.target_col], model_type='sklearn_gb')

        # 融合预测，动态调整权重
        logging.info("\nCombining model predictions...")
        # 计算各模型的验证 RMSE
        lgb_rmse = np.sqrt(mean_squared_error(np.log1p(train_df[self.target_col]), np.log1p(lgb_oof)))
        cat_rmse = np.sqrt(mean_squared_error(np.log1p(train_df[self.target_col]), np.log1p(cat_oof)))
        gb_rmse = np.sqrt(mean_squared_error(np.log1p(train_df[self.target_col]), np.log1p(sklearn_gb_oof)))

        # 权重反比例于 RMSE
        total_inv_rmse = (1 / lgb_rmse) + (1 / cat_rmse) + (1 / gb_rmse)
        lgb_weight = (1 / lgb_rmse) / total_inv_rmse
        cat_weight = (1 / cat_rmse) / total_inv_rmse
        gb_weight = (1 / gb_rmse) / total_inv_rmse

        logging.info(f"Model Weights - LightGBM: {lgb_weight:.2f}, CatBoost: {cat_weight:.2f}, GradientBoosting: {gb_weight:.2f}")

        final_oof = lgb_weight * lgb_oof + cat_weight * cat_oof + gb_weight * sklearn_gb_oof
        final_test = lgb_weight * lgb_test + cat_weight * cat_test + gb_weight * sklearn_gb_test

        # 后处理预测结果，确保在合理范围内
        final_test = np.clip(final_test, train_df[self.target_col].min(), train_df[self.target_col].max())

        # 记录总训练时间
        elapsed_time = time.time() - start_time
        logging.info(f'\nTotal training time: {elapsed_time/60:.2f} minutes')

        # 计算最终验证分数
        final_score = np.sqrt(mean_squared_error(np.log1p(train_df[self.target_col]), np.log1p(final_oof)))
        logging.info(f'Final validation RMSE (log scale): {final_score:.4f}')

        return final_oof, final_test

if __name__ == "__main__":
    # 加载数据
    train_df = pd.read_csv("preprocessing/2024-10-21-silan/train_cleaned.csv")
    test_df = pd.read_csv("preprocessing/2024-10-21-silan/test_cleaned.csv")

    # 处理可能的列名不一致（如空格和特殊字符）
    train_df.columns = [col.strip().replace(' ', '_').replace('&', 'and') for col in train_df.columns]
    test_df.columns = [col.strip().replace(' ', '_').replace('&', 'and') for col in test_df.columns]

    # 初始化和运行模型
    predictor = SGCarPricePredictor(n_folds=5, n_bags=3, seed=42)
    oof_predictions, test_predictions = predictor.fit_predict(train_df, test_df)

    # 创建提交文件
    submission = pd.DataFrame({
        'Id': test_df['Id'] if 'Id' in test_df.columns else range(len(test_predictions)),
        'Predicted': np.round(test_predictions).astype(int)
    })
    submission.to_csv('1025_submission_mixed_optimized.csv', index=False)
    logging.info("Predictions complete. Submission file saved as '1025_submission_mixed_optimized.csv'.")