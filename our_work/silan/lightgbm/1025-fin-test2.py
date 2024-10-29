import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
import logging
import time
from datetime import datetime
import warnings
import optuna
import shap

warnings.filterwarnings('ignore')

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

    def get_model_instance(self, model_name, bag_seed):
        """根据模型名称和种子返回一个新的模型实例"""
        if model_name == 'lightgbm':
            return lgb.LGBMRegressor(
                objective='regression',
                metric='rmse',
                boosting_type='gbdt',
                verbosity=-1,
                learning_rate=0.05,
                num_leaves=31,
                feature_fraction=0.8,
                bagging_fraction=0.7,
                bagging_freq=5,
                max_depth=-1,
                min_child_samples=20,
                lambda_l2=10,
                random_state=bag_seed,
                n_estimators=1000
            )
        elif model_name == 'catboost':
            return CatBoostRegressor(
                iterations=1000,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=10,
                min_data_in_leaf=20,
                random_strength=0.5,
                bagging_temperature=0.2,
                od_type='Iter',
                od_wait=50,
                random_seed=bag_seed,  # 仅使用 random_seed
                verbose=False
            )
        elif model_name == 'sklearn_gb':
            return GradientBoostingRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=5,
                min_samples_split=20,
                min_samples_leaf=15,
                loss='huber',
                random_state=bag_seed
            )
        elif model_name == 'random_forest':
            return RandomForestRegressor(
                n_estimators=1000,
                max_depth=None,
                min_samples_split=10,
                min_samples_leaf=5,
                bootstrap=True,
                random_state=bag_seed,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model type: {model_name}")

    def train_base_models(self, train, test, target_col):
        """训练基模型并生成Oof和测试集预测"""
        oof_preds = {}
        test_preds = {}
        feature_importance = pd.DataFrame()
        base_models = [
            'lightgbm',
            'catboost',
            'sklearn_gb',
            'random_forest'
            # 可以根据需要添加更多基模型
        ]

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        for model_name in base_models:
            logging.info(f"\nTraining base model: {model_name}")
            oof = np.zeros(len(train))
            test_fold_preds = np.zeros(len(test))
            
            for bag in range(self.n_bags):
                bag_seed = self.seed + bag + 1  # 为每个袋装设置不同的种子
                for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
                    start_time = time.time()
                    X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
                    y_train = np.log1p(target_col.iloc[train_idx])
                    y_val = np.log1p(target_col.iloc[val_idx])

                    # 获取新的模型实例
                    model = self.get_model_instance(model_name, bag_seed)

                    # 训练模型
                    if model_name in ['lightgbm', 'catboost']:
                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            # early_stopping_rounds=50,
                            # verbose=False
                        )
                    else:
                        model.fit(X_train, y_train)

                    # 预测
                    oof_fold_pred = model.predict(X_val)
                    oof[val_idx] += np.expm1(oof_fold_pred) / (self.n_folds * self.n_bags)

                    test_fold_pred = model.predict(test)
                    test_fold_preds += np.expm1(test_fold_pred) / (self.n_folds * self.n_bags)

                    # 收集特征重要性
                    if hasattr(model, 'feature_importances_'):
                        importance = pd.DataFrame({
                            'feature': train.columns,
                            'importance': model.feature_importances_,
                            'model': f"{model_name}_bag{bag+1}",
                            'fold': fold + 1
                        })
                        feature_importance = pd.concat([feature_importance, importance], axis=0)

                    elapsed = time.time() - start_time
                    fold_rmse = np.sqrt(mean_squared_error(np.expm1(y_val), oof[val_idx]))
                    logging.info(f'Bag {bag+1} - Fold {fold+1} - {model_name} - RMSE: {fold_rmse:.4f} - Time: {elapsed:.2f}s')

            oof_preds[model_name] = oof
            test_preds[model_name] = test_fold_preds

            # 打印特征重要性
            if model_name in ['lightgbm', 'catboost', 'sklearn_gb', 'random_forest']:
                mean_importance = feature_importance[feature_importance['model'].str.startswith(model_name)].groupby('feature')['importance'].mean().sort_values(ascending=False)
                logging.info(f"\nTop 10 important features for {model_name}:")
                logging.info(mean_importance.head(10))

        return oof_preds, test_preds

    def train_meta_model(self, base_oof, target):
        """训练元模型进行堆叠"""
        logging.info("\nTraining meta-model (Stacking)...")
        meta_train = pd.DataFrame(base_oof)
        meta_test = pd.DataFrame(base_oof).copy()

        meta_model = Ridge(alpha=1.0, random_state=self.seed)
        meta_model.fit(meta_train, np.log1p(target))

        meta_oof_pred = meta_model.predict(meta_train)
        meta_test_pred = meta_model.predict(meta_test)

        # 将预测结果反变换
        meta_oof = np.expm1(meta_oof_pred)
        meta_test_preds = np.expm1(meta_test_pred)

        # 计算 RMSE
        final_score = np.sqrt(mean_squared_error(target, meta_oof))
        logging.info(f'Final validation RMSE (Stacking): {final_score:.4f}')

        # 使用 SHAP 进行特征重要性分析
        logging.info("\nAnalyzing feature importances with SHAP for meta-model...")
        explainer = shap.Explainer(meta_model, meta_train)
        shap_values = explainer(meta_train)
        shap.summary_plot(shap_values, meta_train, show=False)
        plt.savefig('meta_model_shap_summary.png', bbox_inches='tight')
        plt.close()
        logging.info("SHAP summary plot saved as 'meta_model_shap_summary.png'.")

        return meta_oof, meta_test_preds, meta_model

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

        # 训练基模型
        base_oof, base_test = self.train_base_models(train, test, train_df[self.target_col])

        # 训练元模型（Stacking）
        final_oof, final_test, meta_model = self.train_meta_model(base_oof, train_df[self.target_col])

        # 后处理预测结果，确保在合理范围内
        final_test = np.clip(final_test, train_df[self.target_col].min(), train_df[self.target_col].max())

        # 记录总训练时间
        elapsed_time = time.time() - start_time
        logging.info(f'\nTotal training time: {elapsed_time/60:.2f} minutes')

        return final_oof, final_test

    def optimize_hyperparameters(self, train, target_col):
        """使用 Optuna 进行贝叶斯超参数优化"""
        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
                'num_leaves': trial.suggest_int('num_leaves', 20, 50),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'min_child_samples': trial.suggest_int('min_child_samples', 20, 50),
                'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
            }

            model = lgb.LGBMRegressor(**param, random_state=self.seed)
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
            rmse = 0
            for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
                X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
                y_train = np.log1p(target_col.iloc[train_idx])
                y_val = np.log1p(target_col.iloc[val_idx])

                model.fit(X_train, y_train,
                          eval_set=[(X_val, y_val)],
                          early_stopping_rounds=50,
                          verbose=False)

                preds = model.predict(X_val)
                rmse += np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(preds)))

            return rmse / self.n_folds

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)

        logging.info(f'Best trial: {study.best_trial.value}')
        logging.info(f'Best parameters: {study.best_trial.params}')

        return study.best_trial.params

if __name__ == "__main__":
    import matplotlib.pyplot as plt  # 用于保存 SHAP 图

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
    submission.to_csv('1025_submission_stacking_optimized_corrected.csv', index=False)
    logging.info("Predictions complete. Submission file saved as '1025_submission_stacking_optimized_corrected.csv'.")