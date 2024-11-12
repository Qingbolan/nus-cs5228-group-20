# train_xgboost_models.py

import pandas as pd
import numpy as np
import xgboost as xgb
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

# 设置日志（可以选择不在这里设置，统一在 common_utils.py 中设置）
# logging.basicConfig(...)  # 已在 common_utils.py 中设置

def train_evaluate_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Dict[str, Any],
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50
) -> xgb.Booster:
    """
    训练 XGBoost 模型并评估验证集性能。
    """
    # 转换标签为对数空间
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    # 创建 DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train_log)
    dval = xgb.DMatrix(X_val, label=y_val_log)

    evallist = [(dtrain, 'train'), (dval, 'eval')]

    # 训练模型
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evallist,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False  # 关闭训练日志输出
    )

    return model

def train_fold(
    fold: int,
    train_index: np.ndarray,
    val_index: np.ndarray,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, Any],
    num_boost_round: int,
    early_stopping_rounds: int
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
    model = train_evaluate_xgboost(
        X_tr_processed,
        y_tr,
        X_val_processed,
        y_val,
        params,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds
    )

    # 预测验证集
    dval = xgb.DMatrix(X_val_processed)
    fold_predictions = np.expm1(model.predict(dval))  # 还原对数变换

    # 收集特征重要性
    importance = model.get_score(importance_type='gain')
    feature_importance = pd.DataFrame({
        'feature': list(importance.keys()),
        'importance': list(importance.values())
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

def train_xgboost_models(
    train_file_path: str,
    test_file_path: str,
    prediction_output_path: str,
    model_save_path: str = 'xgboost_models.pkl',
    n_splits: int = 5,
    n_jobs: int = -1,
    num_boost_round: int = 1000,
    learning_rate: float = 0.05,
    early_stopping_rounds: int = 50,
    use_gpu: bool = False
):
    """
    训练 XGBoost 模型，使用 K-Fold 交叉验证，并在测试集上进行预测。
    预测结果保存到指定路径，同时保存训练好的模型和预处理器。
    """

    try:
        np.random.seed(42)
        logging.info("开始加载训练数据...")
        X_train, y_train = load_data(train_file_path)

        logging.info("\n目标变量 (price) 统计信息:")
        logging.info(y_train.describe())

        # 初始化 K-Fold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        oof_predictions = np.zeros(len(X_train))
        feature_importance_list = []
        models = []

        # XGBoost 参数
        xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': 'gbtree',
            'eta': learning_rate,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0,    # L1正则化
            'reg_lambda': 1,    # L2正则化
            'seed': 42,
            'verbosity': 0      # 关闭XGBoost日志
        }

        # 如果使用GPU，调整参数
        if use_gpu:
            xgb_params['tree_method'] = 'gpu_hist'
            logging.info("启用GPU加速进行训练")
        else:
            xgb_params['tree_method'] = 'hist'
            logging.info("使用CPU进行训练")

        start_time = time.time()

        # 并行训练 K-Fold
        results = Parallel(n_jobs=n_jobs)(
            delayed(train_fold)(
                fold=fold,
                train_index=train_index,
                val_index=val_index,
                X_train=X_train,
                y_train=y_train,
                params=xgb_params,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds
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
        predict_xgboost(
            model_save_path=model_save_path,
            test_file_path=test_file_path,
            prediction_output_path=prediction_output_path,
            model_type='xgboost'
        )

    except Exception as e:
        logging.error(f"训练和预测过程中出现错误: {str(e)}")
        raise

def predict_xgboost(
    model_save_path: str,
    test_file_path: str,
    prediction_output_path: str,
    model_type: str = 'xgboost'
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

            # 创建 DMatrix
            dtest = xgb.DMatrix(X_test_processed)

            # 预测并还原对数变换
            preds = np.expm1(model.predict(dtest))
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
    prediction_file = 'submission_xgboost.csv'

    train_xgboost_models(
        train_file_path=train_file,
        test_file_path=test_file,
        prediction_output_path=prediction_file,
        use_gpu=True
    )
