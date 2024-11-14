# train_catboost_models.py

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import time
import logging
from typing import Tuple, List, Dict, Any, Optional
from joblib import Parallel, delayed
from final_report.approach1.without_cluster.models.common_utils import (
    load_data,
    preprocess_features,
    post_process_predictions,
    verify_saved_model,
    save_model
)

def train_evaluate_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Dict[str, Any]
) -> CatBoostRegressor:
    """
    训练 CatBoostRegressor 模型，并评估验证集性能。
    """
    model = CatBoostRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
        verbose=False,
        early_stopping_rounds=100
    )

    # 计算验证集指标
    val_predictions = model.predict(X_val)
    val_mse = mean_squared_error(y_val, val_predictions)
    val_r2 = r2_score(y_val, val_predictions)

    logging.info(f"验证集 RMSE: {np.sqrt(val_mse):.4f}")
    logging.info(f"验证集 R2: {val_r2:.4f}")

    return model

def train_fold(
    fold: int,
    train_index: np.ndarray,
    val_index: np.ndarray,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, Any]
) -> Tuple[int, Dict[str, Any], np.ndarray, np.ndarray, pd.DataFrame]:
    """
    训练单个折，并返回相关信息。
    """
    logging.info(f"\n开始训练第 {fold} 折模型")

    X_tr, X_val = X_train.iloc[train_index].copy(), X_train.iloc[val_index].copy()
    y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

    # 预处理
    X_tr_processed, num_imputer, scaler = preprocess_features(X_tr, y_tr)
    X_val_processed, _, _ = preprocess_features(
        X_val,
        y_val,
        num_imputer=num_imputer,
        scaler=scaler
    )

    # 训练模型
    model = train_evaluate_catboost(
        X_tr_processed,
        y_tr,
        X_val_processed,
        y_val,
        params
    )

    # 预测验证集
    fold_predictions = model.predict(X_val_processed)

    # 收集特征重要性
    importance = model.get_feature_importance()
    feature_importance = pd.DataFrame({
        'feature': X_tr_processed.columns,
        'importance': importance
    })

    # 保存模型和预处理器
    model_dict = {
        'model': model,
        'preprocessors': {
            'num_imputer': num_imputer,
            'scaler': scaler
        }
    }

    return fold, model_dict, val_index, fold_predictions, feature_importance

def train_catboost_models(
    train_file_path: str,
    test_file_path: str,
    prediction_output_path: str,
    model_save_path: str = 'catboost_models.pkl',
    n_splits: int = 5,
    n_jobs: int = -1,
    cb_params: Dict[str, Any] = None
):
    """
    训练 CatBoostRegressor 模型，使用 K-Fold 交叉验证，并在测试集上进行预测。
    预测结果保存到指定路径，同时保存训练好的模型和预处理器。
    """

    try:
        np.random.seed(42)
        logging.info("开始加载训练数据...")
        X_train, y_train = load_data(train_file_path)

        logging.info("\n目标变量 (price) 统计信息:")
        logging.info(y_train.describe())

        # 初始化 K-Fold
        sample_count = len(y_train)
        if sample_count < 5:
            logging.warning(f"训练数据样本数量为 {sample_count}，小于 5，调整交叉验证的折数为 {sample_count}")
            n_splits = 2
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        oof_predictions = np.zeros(len(X_train))
        feature_importance_list = []
        models = []

        # CatBoost 参数
        if cb_params is None:
            cb_params = {
                'iterations': 5000,
                'learning_rate': 0.1,
                'depth': 6,
                'l2_leaf_reg': 10,
                'min_data_in_leaf': 20,
                'random_strength': 0.5,
                'bagging_temperature': 0.2,
                'od_type': 'Iter',
                'od_wait': 100,
                'random_seed': 42,
                'verbose': False,
                'task_type': 'CPU',
                'allow_writing_files': False
            }

        start_time = time.time()

        # 并行训练 K-Fold
        results = Parallel(n_jobs=n_jobs)(
            delayed(train_fold)(
                fold=fold,
                train_index=train_index,
                val_index=val_index,
                X_train=X_train,
                y_train=y_train,
                params=cb_params
            )
            for fold, (train_index, val_index) in enumerate(kf.split(X_train), 1)
        )

        for result in results:
            fold, model_dict, val_index, fold_predictions, feature_importance = result
            logging.info(f"第 {fold} 折训练完成")

            # 赋值到 oof_predictions
            oof_predictions[val_index] = fold_predictions

            feature_importance_list.append(feature_importance)
            models.append(model_dict)

        elapsed_time = time.time() - start_time
        logging.info(f"\n总训练时间: {elapsed_time / 60:.2f} 分钟")

        # 评估 OOF 预测
        oof_predictions = post_process_predictions(oof_predictions)
        oof_mse = mean_squared_error(y_train, oof_predictions)
        oof_r2 = r2_score(y_train, oof_predictions)
        logging.info(f"Out-of-fold RMSE: {np.sqrt(oof_mse):.4f}")
        logging.info(f"Out-of-fold R2: {oof_r2:.4f}")

        # 特征重要性分析
        feature_importance = pd.concat(feature_importance_list).groupby('feature').mean().sort_values('importance', ascending=False)
        logging.info("\nTop 10 重要特征:")
        logging.info(feature_importance.head(10))

        # 保存模型和预处理器
        save_model(model_save_path, models, feature_importance)

        # 验证保存的模型
        if not verify_saved_model(model_save_path):
            raise RuntimeError("模型验证失败，保存的模型不完整或损坏")

        # 推理
        predict_catboost(
            model_save_path=model_save_path,
            test_file_path=test_file_path,
            prediction_output_path=prediction_output_path,
            model_type='catboost'
        )

    except Exception as e:
        logging.error(f"训练和预测过程中出现错误: {str(e)}")
        raise

def predict_catboost(
    model_save_path: str,
    test_file_path: str,
    prediction_output_path: str,
    model_type: str = 'catboost'
):
    """
    使用保存的模型对测试数据进行预测。
    
    Args:
        model_save_path (str): 保存的模型路径。
        test_file_path (str): 测试数据文件路径。
        prediction_output_path (str): 预测结果保存路径。
        model_type (str): 模型类型，用于选择预测方法。
    """
    try:
        # 加载模型
        with open(model_save_path, 'rb') as f:
            loaded_model = pickle.load(f)

        models = loaded_model['models']
        logging.info(f"加载了 {len(models)} 个模型")

        # 加载测试数据
        X_test, _ = load_data(test_file_path)

        final_predictions = np.zeros(len(X_test))

        # 对每个模型进行预测并取平均
        for i, model_dict in enumerate(models, 1):
            logging.info(f"使用模型 {i} 进行预测")
            model = model_dict['model']
            preprocessors = model_dict['preprocessors']

            # 预处理测试数据
            X_test_processed, _, _ = preprocess_features(
                X_test,
                y=None,
                num_imputer=preprocessors['num_imputer'],
                scaler=preprocessors['scaler']
            )

            # 预测
            preds = model.predict(X_test_processed)
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
        logging.info(f"预测完成。提交文件已保存为 '{prediction_output_path}'")

        # 输出预测统计信息
        logging.info("\n预测统计信息:")
        logging.info(f"最小值: {final_predictions.min():.2f}")
        logging.info(f"最大值: {final_predictions.max():.2f}")
        logging.info(f"均值: {final_predictions.mean():.2f}")
        logging.info(f"中位数: {np.median(final_predictions):.2f}")

    except Exception as e:
        logging.error(f"预测过程中出现错误: {str(e)}")
        raise

if __name__ == '__main__':
    # 示例用法
    train_file = 'preprocessing/2024-10-21-silan/train_cleaned.csv'
    test_file = 'preprocessing/2024-10-21-silan/test_cleaned.csv'
    prediction_file = '10-27-release_catboost.csv'

    train_catboost_models(
        train_file_path=train_file,
        test_file_path=test_file,
        prediction_output_path=prediction_file
    )
