import os
import pickle
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from data.rmse_calculator import calculate_rmse

# 导入已有的模型训练函数
from final_report.approach1.without_cluster.models.xgboost import train_xgboost_models
from final_report.approach1.without_cluster.models.lightgbm import train_lightgbm_models
from final_report.approach1.without_cluster.models.catboost import train_catboost_models
from final_report.approach1.without_cluster.models.gradientboost import train_gradient_boosting_models

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

load_dotenv()

def define_mistake_categories(custom_categories: List[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
    """
    定义误差绝对值的分类区间。
    """
    if custom_categories:
        return custom_categories
    else:
        # 默认分类区间
        categories = [
            (0, 8000),
            (8001, 9999999)  # 确保包括所有可能的误差值
        ]
        return categories

def assign_mistake_category(mistake: float, categories: List[Tuple[int, int]]) -> int:
    """
    根据误差的绝对值将其分配到相应的分类区间。
    """
    abs_mistake = abs(mistake)
    for idx, (lower, upper) in enumerate(categories):
        if lower <= abs_mistake <= upper:
            return idx
    return len(categories)  # 超出定义区间的分类

def load_model(model_save_path: str) -> Dict[str, Any]:
    """
    加载保存的模型。
    """
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"模型文件未找到: {model_save_path}")

    with open(model_save_path, 'rb') as f:
        model = pickle.load(f)

    logging.info(f"成功加载模型: {model_save_path}")
    return model

def perform_inference(
    model: Dict[str, Any],
    data_file_path: str
) -> pd.DataFrame:
    """
    使用加载的模型对指定的数据集进行推理，生成 inference_price 和 mistake 列。
    """
    from final_report.approach1.without_cluster.models.common_utils import load_data, preprocess_features, post_process_predictions

    X, y = load_data(data_file_path)
    logging.info(f"开始对数据集 {data_file_path} 进行推理。数据形状: {X.shape}")

    model_dict = model['models'][0]  # 使用单个模型
    logging.info(f"使用模型进行推理")
    trained_model = model_dict['model']
    preprocessors = model_dict['preprocessors']

    # **确保删除 'Id' 列，避免将其作为特征**
    if 'Id' in X.columns:
        X = X.drop(columns=['Id'])
        logging.info("在特征中删除 'Id' 列")

    # 预处理
    X_processed, _, _ = preprocess_features(
        X,
        y=None,
        num_imputer=preprocessors['num_imputer'],
        scaler=preprocessors['scaler']
    )

    # 预测
    if isinstance(trained_model, dict):
        # 如果模型保存为字典结构
        trained_model = trained_model['model']

    import xgboost as xgb
    dtest = xgb.DMatrix(X_processed)
    preds_log = trained_model.predict(dtest)
    preds = np.expm1(preds_log)

    # 后处理
    final_predictions = post_process_predictions(preds)

    # 生成结果数据框
    results = X.copy()
    results['inference_price'] = final_predictions
    if y is not None:
        results['price'] = y
        results['mistake'] = results['price'] - results['inference_price']
    else:
        logging.warning("没有目标变量 'price'，无法计算 'mistake'")
        results['mistake'] = None

    logging.info("推理完成，生成 inference_price 和 mistake 列。")
    return results

def retrain_models_on_subsets(
    data: pd.DataFrame,
    model_name: str,
    output_dir: str,
    test_file_path: str  # 添加参数，传入测试集路径
) -> Dict[int, str]:
    """
    针对每个误差类别的子集，调用对应的模型训练函数重新训练模型。

    Returns:
        一个字典，键为类别 ID，值为对应模型的保存路径。
    """
    models = {}
    for category_id in sorted(data['mistake_category'].unique()):
        subset_data = data[data['mistake_category'] == category_id].copy()
        logging.info(f"重新训练类别 {category_id} 的模型，样本数量: {len(subset_data)}")

        # 删除不相关的列
        subset_data = subset_data.drop(columns=['inference_price', 'mistake', 'mistake_category', 'mistake_abs'], errors='ignore')

        # **确保删除 'Id' 列，避免将其作为特征**
        if 'Id' in subset_data.columns:
            subset_data = subset_data.drop(columns=['Id'])
            logging.info("在训练数据中删除 'Id' 列")

        # 保存子集数据到临时文件，确保不保存索引
        subset_train_file = os.path.join(output_dir, f"train_data_category_{category_id}.csv")
        subset_data.to_csv(subset_train_file, index=False)  # 确保不保存索引

        # 定义模型保存路径
        model_save_path = os.path.join(output_dir, f"{model_name}_model_category_{category_id}.pkl")

        # 调用对应的模型训练函数
        if model_name == 'xgboost':
            # 传入实际的测试集路径，避免传递 None
            train_xgboost_models(
                train_file_path=subset_train_file,
                test_file_path=test_file_path,
                prediction_output_path=None,  # 不需要预测输出
                model_save_path=model_save_path
            )
        elif model_name == 'lightgbm':
            train_lightgbm_models(
                train_file_path=subset_train_file,
                test_file_path=test_file_path,
                prediction_output_path=None,  # 不需要预测输出
                model_save_path=model_save_path
            )
        elif model_name == 'catboost':
            train_catboost_models(
                train_file_path=subset_train_file,
                test_file_path=test_file_path,
                prediction_output_path=None,  # 不需要预测输出
                model_save_path=model_save_path
            )
        elif model_name == 'gradient_boosting':
            train_gradient_boosting_models(
                train_file_path=subset_train_file,
                test_file_path=test_file_path,
                prediction_output_path=None,  # 不需要预测输出
                model_save_path=model_save_path
            )
        else:
            raise ValueError(f"不支持的模型名称: {model_name}")

        logging.info(f"类别 {category_id} 的模型已保存到 {model_save_path}")
        models[category_id] = model_save_path

        # 删除临时文件
        os.remove(subset_train_file)

    return models

def train_mistake_category_classifier(
    data: pd.DataFrame,
    output_dir: str,
    classifier_name: str = 'xgboost',
    classifier_save_path: str = None
) -> Any:
    """
    训练一个分类器来预测误差类别。

    Returns:
        训练好的分类器
    """
    from sklearn.preprocessing import LabelEncoder
    from final_report.approach1.without_cluster.models.common_utils import preprocess_features

    # 提取特征和误差类别
    X = data.drop(columns=['price', 'inference_price', 'mistake', 'mistake_abs', 'mistake_category'], errors='ignore')

    # **确保删除 'Id' 列，避免将其作为特征**
    if 'Id' in X.columns:
        X = X.drop(columns=['Id'])
        logging.info("在训练分类器时删除 'Id' 列")

    y = data['mistake_category']

    # 对类别进行编码
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    logging.info(f"误差类别编码映射: {class_mapping}")

    # 预处理
    num_imputer = None
    scaler = None
    X_processed, num_imputer, scaler = preprocess_features(
        X.copy(),
        y=None,
        num_imputer=None,
        scaler=None
    )

    # 训练分类器
    if classifier_name == 'xgboost':
        import xgboost as xgb
        classifier = xgb.XGBClassifier(
            max_depth=6,
            n_estimators=100,
            learning_rate=0.1,
            objective='multi:softmax',
            num_class=len(label_encoder.classes_),
            verbosity=0
        )
        classifier.fit(X_processed, y_encoded)
    else:
        raise ValueError(f"不支持的分类器名称: {classifier_name}")

    # 保存分类器
    if not classifier_save_path:
        classifier_save_path = os.path.join(output_dir, f"{classifier_name}_mistake_category_classifier.pkl")

    with open(classifier_save_path, 'wb') as f:
        pickle.dump({
            'classifier': classifier,
            'label_encoder': label_encoder,
            'num_imputer': num_imputer,
            'scaler': scaler
        }, f)

    logging.info(f"误差类别分类器已保存到 {classifier_save_path}")

    return classifier

def predict_with_retrained_models(
    test_data: pd.DataFrame,
    models: Dict[int, str],
    model_name: str,
    classifier_path: str
) -> pd.DataFrame:
    """
    使用训练好的分类器为测试数据分配误差类别，并使用对应的模型进行预测。

    Returns:
        包含预测结果的 DataFrame
    """
    from final_report.approach1.without_cluster.models.common_utils import preprocess_features, post_process_predictions

    # 加载分类器
    with open(classifier_path, 'rb') as f:
        classifier_data = pickle.load(f)
    classifier = classifier_data['classifier']
    label_encoder = classifier_data['label_encoder']
    num_imputer = classifier_data['num_imputer']
    scaler = classifier_data['scaler']

    # 准备测试数据
    if 'Unnamed: 0' in test_data.columns:
        logging.info("删除 'Unnamed: 0' 列")
        test_data = test_data.drop(columns=['Unnamed: 0'])

    # 检查是否有 'Id' 列，如果没有，用索引创建
    if 'Id' not in test_data.columns:
        logging.warning("'Id' 列不存在，使用索引创建 'Id' 列")
        test_data['Id'] = test_data.index

    # **在特征中删除 'Id' 列，避免将其作为特征**
    test_data_clean = test_data.drop(columns=['price', 'inference_price', 'mistake', 'mistake_category', 'mistake_abs', 'Id'], errors='ignore')
    test_indices = test_data.index

    # 预处理测试数据
    X_test_processed, _, _ = preprocess_features(
        test_data_clean.copy(),
        y=None,
        num_imputer=num_imputer,
        scaler=scaler
    )

    # 预测误差类别
    y_pred_encoded = classifier.predict(X_test_processed)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    test_data['mistake_category'] = y_pred

    # 根据预测的类别分组
    prediction_results = pd.DataFrame(index=test_indices)
    final_predictions = np.zeros(len(test_data))  # 用于统计预测值

    for category_id in sorted(test_data['mistake_category'].unique()):
        category_data = test_data[test_data['mistake_category'] == category_id]

        if category_id not in models:
            logging.warning(f"类别 {category_id} 的模型不存在，跳过该类别的预测")
            continue

        model_path = models[category_id]
        logging.info(f"使用类别 {category_id} 的模型进行预测，模型路径：{model_path}")

        # 加载模型
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        model_dict = model['models'][0]
        trained_model = model_dict['model']
        preprocessors = model_dict['preprocessors']

        # 提取特征
        X_category = category_data.drop(columns=['price', 'inference_price', 'mistake', 'mistake_category', 'mistake_abs', 'Id'], errors='ignore')
        indices = X_category.index

        # 预处理
        X_category_processed, _, _ = preprocess_features(
            X_category.copy(),
            y=None,
            num_imputer=preprocessors['num_imputer'],
            scaler=preprocessors['scaler']
        )

        # 预测
        if model_name == 'xgboost':
            import xgboost as xgb
            dtest = xgb.DMatrix(X_category_processed)
            preds_log = trained_model.predict(dtest)
            preds = np.expm1(preds_log)
        else:
            raise ValueError(f"不支持的模型名称: {model_name}")

        # 后处理
        preds = post_process_predictions(preds)

        # 保存预测结果
        prediction_results_category = pd.DataFrame({'Predicted': preds}, index=indices)
        prediction_results = prediction_results.combine_first(prediction_results_category)

        # 保存预测值用于统计
        final_predictions[indices] = preds

    # 检查是否有未预测的样本
    missing_indices = test_indices.difference(prediction_results.index)
    if len(missing_indices) > 0:
        logging.warning(f"{len(missing_indices)} 个测试样本未被分类器分配到已知的误差类别，将使用默认模型进行预测")

        # 使用默认模型（可以选择一个已有的模型，或者重新训练一个全量模型）
        default_model_path = models[next(iter(models))]  # 使用第一个模型
        logging.info(f"使用默认模型进行预测，模型路径：{default_model_path}")

        # 加载模型
        with open(default_model_path, 'rb') as f:
            model = pickle.load(f)
        model_dict = model['models'][0]
        trained_model = model_dict['model']
        preprocessors = model_dict['preprocessors']

        # 提取特征
        X_missing = test_data.loc[missing_indices].drop(columns=['price', 'inference_price', 'mistake', 'mistake_category', 'mistake_abs', 'Id'], errors='ignore')

        # 预处理
        X_missing_processed, _, _ = preprocess_features(
            X_missing.copy(),
            y=None,
            num_imputer=preprocessors['num_imputer'],
            scaler=preprocessors['scaler']
        )

        # 预测
        if model_name == 'xgboost':
            import xgboost as xgb
            dtest = xgb.DMatrix(X_missing_processed)
            preds_log = trained_model.predict(dtest)
            preds = np.expm1(preds_log)
        else:
            raise ValueError(f"不支持的模型名称: {model_name}")

        # 后处理
        preds = post_process_predictions(preds)

        # 保存预测结果
        prediction_results_missing = pd.DataFrame({'Predicted': preds}, index=missing_indices)
        prediction_results = prediction_results.combine_first(prediction_results_missing)

        # 保存预测值用于统计
        final_predictions[missing_indices] = preds

    # 重新设置索引，确保与测试集顺序一致
    prediction_results = prediction_results.reindex(test_indices)

    # 添加 'Id' 列，确保 submission 文件有 'Id' 和 'Predicted' 两列
    prediction_results['Id'] = test_data.loc[test_indices, 'Id'].values
    prediction_results = prediction_results[['Id', 'Predicted']]

    # 输出预测统计信息
    logging.info("\n预测统计信息:")
    logging.info(f"最小值: {final_predictions.min():.2f}")
    logging.info(f"最大值: {final_predictions.max():.2f}")
    logging.info(f"均值: {final_predictions.mean():.2f}")
    logging.info(f"中位数: {np.median(final_predictions):.2f}")

    return prediction_results

def main():
    # 定义文件路径和参数
    train_file_path = os.getenv('FINAL_REPORT_TRAIN_DATA')
    test_file_path = os.getenv('FINAL_REPORT_TEST_DATA')
    model_save_path = os.getenv('FINAL_REPORT_APPROACH_one_WITHOUT_cluster_XGBoost_WEIGHT_PATH')
    submission_output_path = os.getenv('FINAL_REPORT_APPROACH_one_SUBMISSION_PATH')
    error_analysis_output_dir = 'final_report/approach1/without_cluster/error_analysis'
    model_name = 'xgboost'  # 可以根据需要更改为 'lightgbm' 等

    os.makedirs(error_analysis_output_dir, exist_ok=True)

    custom_categories = [
        (0, 8000),
        (8001, 9999999)
    ]

    # 加载已有模型
    model = load_model(model_save_path)

    # 对训练数据进行推理，计算误差
    inference_results = perform_inference(model, train_file_path)

    # 定义误差分类区间并分配类别
    categories = define_mistake_categories(custom_categories)
    inference_results['mistake_abs'] = inference_results['mistake'].abs()
    inference_results['mistake_category'] = inference_results['mistake'].apply(
        lambda x: assign_mistake_category(x, categories)
    )

    # 删除不相关的列
    inference_results_clean = inference_results.drop(columns=['inference_price', 'mistake', 'mistake_abs'], errors='ignore')

    # 保存处理后的训练数据，确保不保存索引
    processed_train_file = os.path.join(error_analysis_output_dir, "processed_train_data.csv")
    inference_results_clean.to_csv(processed_train_file, index=False)

    # 重新训练每个子集的模型，调用对应的训练函数
    models = retrain_models_on_subsets(
        data=inference_results_clean,
        model_name=model_name,
        output_dir=error_analysis_output_dir,
        test_file_path=test_file_path  # 传入测试集路径，避免传递 None
    )

    # 训练误差类别分类器
    classifier_save_path = os.path.join(error_analysis_output_dir, f"{model_name}_mistake_category_classifier.pkl")
    train_mistake_category_classifier(
        data=inference_results,
        output_dir=error_analysis_output_dir,
        classifier_name=model_name,
        classifier_save_path=classifier_save_path
    )

    # 读取测试数据
    test_data = pd.read_csv(test_file_path)
    if 'Unnamed: 0' in test_data.columns:
        logging.info("删除 'Unnamed: 0' 列")
        test_data = test_data.drop(columns=['Unnamed: 0'])

    # 使用分类器和对应的模型对测试集进行预测
    prediction_results = predict_with_retrained_models(
        test_data=test_data,
        models=models,
        model_name=model_name,
        classifier_path=classifier_save_path
    )

    # 保存预测结果，保持与原始数据顺序一致
    prediction_results.to_csv(submission_output_path, index=False)
    logging.info(f"最终的预测结果已保存到 {submission_output_path}")

    # 如果测试集有真实值，可以计算 RMSE
    rmse = calculate_rmse(submission_output_path)
    logging.info(f"预测结果的 RMSE: {rmse}")

    logging.info("模型训练和预测流程完成。")

if __name__ == '__main__':
    main()