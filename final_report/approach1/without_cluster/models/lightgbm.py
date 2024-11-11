import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import pickle
import time
from category_encoders import TargetEncoder
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train_lightgbm_models(train_file_path, test_file_path, prediction_output_path):
    """
    训练 LightGBM 模型，使用 K-Fold 交叉验证，并在测试集上进行预测。
    预测结果保存到指定路径，同时保存训练好的模型和预处理器。

    参数:
    - train_file_path (str): 训练集 CSV 文件路径。
    - test_file_path (str): 测试集 CSV 文件路径。
    - prediction_output_path (str): 预测结果保存的 CSV 文件路径。
    """

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
                            target_encode_cols=['make', 'model'],
                            encoding_smoothing=1.0):
        X = X.copy()

        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns

        # 处理数值特征缺失值
        if num_imputer is None:
            num_imputer = SimpleImputer(strategy='median')
            X[numeric_features] = pd.DataFrame(num_imputer.fit_transform(X[numeric_features]),
                                               columns=numeric_features,
                                               index=X.index)
        else:
            X[numeric_features] = pd.DataFrame(num_imputer.transform(X[numeric_features]),
                                               columns=numeric_features,
                                               index=X.index)

        # 处理分类特征缺失值
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

            # 其他分类特征进行独热编码
            other_categorical = [col for col in categorical_features if col not in target_encode_features]
            if len(other_categorical) > 0:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_features = encoder.fit_transform(X[other_categorical])
                encoded_feature_names = encoder.get_feature_names_out(other_categorical)
                encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=X.index)
                X = pd.concat([X, encoded_df], axis=1)
                X = X.drop(columns=other_categorical)

        return X, num_imputer, cat_imputer, target_encoder, scaler

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
            # verbose_eval=100
        )

        return model

    def post_process_predictions(predictions, min_price=700, max_price=2900000):
        return np.clip(predictions, min_price, max_price)

    # 开始主逻辑
    np.random.seed(42)

    # 加载训练数据
    X_train, y_train = load_and_preprocess_data(train_file_path)

    logging.info("目标变量 (price) 统计信息:")
    logging.info(y_train.describe())

    # 初始化 K-Fold
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    oof_predictions = np.zeros(len(X_train))
    feature_importance_list = []
    models = []

    # LightGBM 参数
    params = {
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

    start_time = time.time()

    for fold, (train_index, val_index) in enumerate(kf.split(X_train), 1):
        logging.info(f"Fold {fold}")

        X_tr, X_val = X_train.iloc[train_index].copy(), X_train.iloc[val_index].copy()
        y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

        # 预处理
        X_tr_processed, num_imputer, cat_imputer, target_encoder, scaler = preprocess_features(X_tr, y_tr)
        X_val_processed, _, _, _, _ = preprocess_features(X_val, y_val, num_imputer, cat_imputer, target_encoder,
                                                          scaler)

        # 训练模型
        model = train_evaluate_lightgbm(X_tr_processed, y_tr, X_val_processed, y_val, params)

        # 预测验证集
        fold_predictions = np.expm1(model.predict(X_val_processed, num_iteration=model.best_iteration))
        oof_predictions[val_index] = fold_predictions

        # 收集特征重要性
        importance = model.feature_importance(importance_type='gain')
        feature_importance = pd.DataFrame({'feature': X_tr_processed.columns, 'importance': importance})
        feature_importance_list.append(feature_importance)

        # 保存模型和预处理器
        models.append({
            'model': model,
            'preprocessors': {
                'num_imputer': num_imputer,
                'cat_imputer': cat_imputer,
                'target_encoder': target_encoder,
                'scaler': scaler
            }
        })

    elapsed_time = time.time() - start_time
    logging.info(f"\n总训练时间: {elapsed_time / 60:.2f} 分钟")

    # 评估
    oof_predictions = post_process_predictions(oof_predictions)
    oof_mse = mean_squared_error(y_train, oof_predictions)
    oof_r2 = r2_score(y_train, oof_predictions)
    logging.info(f"Out-of-fold RMSE: {np.sqrt(oof_mse):.4f}")
    logging.info(f"Out-of-fold R2: {oof_r2:.4f}")

    # 特征重要性
    feature_importance = pd.concat(feature_importance_list).groupby('feature').mean().sort_values('importance',
                                                                                                  ascending=False)
    logging.info("\nTop 10 重要特征:")
    logging.info(feature_importance.head(10))

    # 保存模型和预处理器
    with open('lightgbm_models.pkl', 'wb') as f:
        pickle.dump({
            'models': models,
            'feature_importance': feature_importance
        }, f)
    logging.info("模型和预处理器已保存到 'lightgbm_models.pkl'.")

    # 加载测试数据
    X_test, _ = load_and_preprocess_data(test_file_path)

    final_predictions = np.zeros(len(X_test))

    # 对每个模型进行预测并取平均
    for i, model_dict in enumerate(models, 1):
        logging.info(f"Predicting with model {i}")
        model = model_dict['model']
        preprocessors = model_dict['preprocessors']

        # 预处理测试数据
        X_test_processed, _, _, _, _ = preprocess_features(X_test, y=None,
                                                           num_imputer=preprocessors['num_imputer'],
                                                           cat_imputer=preprocessors['cat_imputer'],
                                                           target_encoder=preprocessors['target_encoder'],
                                                           scaler=preprocessors['scaler'])

        preds = np.expm1(model.predict(X_test_processed, num_iteration=model.best_iteration))
        final_predictions += preds

    # 取平均
    final_predictions /= len(models)

    # 后处理预测值
    final_predictions = post_process_predictions(final_predictions)

    # 保存预测结果
    submission = pd.DataFrame({
        'Id': range(len(final_predictions)),
        'Predicted': np.round(final_predictions).astype(int)
    })

    submission.to_csv(prediction_output_path, index=False)
    logging.info(f"预测完成。提交文件已保存为 '{prediction_output_path}'.")

    # 输出预测统计信息
    logging.info("\n预测统计信息:")
    logging.info(f"最小值: {final_predictions.min()}")
    logging.info(f"最大值: {final_predictions.max()}")
    logging.info(f"均值: {final_predictions.mean()}")
    logging.info(f"中位数: {np.median(final_predictions)}")


if __name__ == '__main__':
    train_file = 'preprocessing/2024-10-21-silan/train_cleaned.csv'
    test_file = 'preprocessing/2024-10-21-silan/test_cleaned.csv'
    prediction_file = './10-27-release_lightgbm.csv'

    train_lightgbm_models(train_file, test_file, prediction_file)
