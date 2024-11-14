# common_utils.py

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pickle
import logging
from typing import Tuple, List, Dict, Any, Optional
import os

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('common_utils.log'),
        logging.StreamHandler()
    ]
)

def load_data(file_path: str) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    加载数据并进行初步处理。
    
    Args:
        file_path (str): 数据文件路径。
        
    Returns:
        Tuple[pd.DataFrame, Optional[pd.Series]]: 特征数据和目标变量。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件未找到: {file_path}")

    try:
        data = pd.read_csv(file_path)
        logging.info(f"成功加载数据文件: {file_path}. 数据形状: {data.shape}")
    except Exception as e:
        logging.error(f"加载数据文件时出错: {file_path} 错误信息: {str(e)}")
        raise

    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns='Unnamed: 0')

    X = data.drop('price', axis=1) if 'price' in data.columns else data
    y = data['price'] if 'price' in data.columns else None

    logging.info(f"特征列: {X.columns.tolist()}")
    if y is not None:
        logging.info(f"目标变量形状: {y.shape}")
        logging.info(f"价格范围: {y.min():.2f} 到 {y.max():.2f}")

    return X, y

def preprocess_features(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    num_imputer: Optional[SimpleImputer] = None,
    scaler: Optional[StandardScaler] = None,
    encoding_smoothing: float = 1.0
) -> Tuple[pd.DataFrame, SimpleImputer, StandardScaler]:
    """
    预处理特征，包括填补缺失值和标准化。
    
    Args:
        X (pd.DataFrame): 特征数据。
        y (Optional[pd.Series]): 目标变量。
        num_imputer (Optional[SimpleImputer]): 数值特征填充器。
        scaler (Optional[StandardScaler]): 标准化器。
        encoding_smoothing (float): 目标编码平滑参数（保留，虽然不使用）。
        
    Returns:
        Tuple[pd.DataFrame, SimpleImputer, StandardScaler]: 预处理后的数据及预处理器。
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X 必须是 pandas DataFrame")

    X = X.copy()
    logging.info("开始特征预处理")

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # 确保 'Unnamed: 0' 列被删除
    if 'Unnamed: 0' in X.columns:
        logging.info("删除 'Unnamed: 0' 列")
        X = X.drop(columns=['Unnamed: 0'])
        
    logging.info(f"数值特征: {numeric_features}")

    # 处理数值特征缺失值
    if numeric_features:
        if num_imputer is None:
            num_imputer = SimpleImputer(strategy='median')
            X[numeric_features] = num_imputer.fit_transform(X[numeric_features])
            logging.info("数值特征缺失值已使用中位数填充")
        else:
            X[numeric_features] = num_imputer.transform(X[numeric_features])
            logging.info("使用现有的数值特征填充器填充数值特征缺失值")

    # 标准化特定的数值特征
    # columns_to_standardize = ['curb_weight', 'power', 'engine_cap', 'depreciation']
    columns_to_standardize = []
    columns_to_standardize = [col for col in columns_to_standardize if col in numeric_features]

    if columns_to_standardize:
        if scaler is None:
            scaler = StandardScaler()
            X[columns_to_standardize] = scaler.fit_transform(X[columns_to_standardize])
            logging.info(f"数值特征 {columns_to_standardize} 已标准化")
        else:
            X[columns_to_standardize] = scaler.transform(X[columns_to_standardize])
            logging.info(f"使用现有的标准化器标准化数值特征 {columns_to_standardize}")
    else:
        logging.info("没有需要标准化的数值特征")

    logging.info(f"完成特征预处理。最终数据形状: {X.shape}")
    return X, num_imputer, scaler

def post_process_predictions(
    predictions: np.ndarray,
    min_price: float = 700,
    max_price: float = 2900000
) -> np.ndarray:
    """
    后处理预测结果，限制价格范围。
    
    Args:
        predictions (np.ndarray): 预测结果。
        min_price (float): 最小价格。
        max_price (float): 最大价格。
        
    Returns:
        np.ndarray: 后处理后的预测结果。
    """
    return np.clip(predictions, min_price, max_price)

def verify_saved_model(model_path: str) -> bool:
    """
    验证保存的模型的完整性和正确性。
    
    Args:
        model_path (str): 模型保存路径。
        
    Returns:
        bool: 是否验证成功。
    """
    try:
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)

        required_keys = ['models', 'feature_importance']
        missing_keys = [key for key in required_keys if key not in loaded_model]

        if missing_keys:
            raise ValueError(f"保存的模型缺少必要的键: {missing_keys}")

        logging.info("模型验证成功")
        return True
    except Exception as e:
        logging.error(f"模型验证失败: {str(e)}")
        return False

def save_model(model_save_path: str, models: List[Dict[str, Any]], feature_importance: pd.DataFrame):
    """
    保存训练好的模型和特征重要性。
    
    Args:
        model_save_path (str): 模型保存路径。
        models (List[Dict[str, Any]]): 训练好的模型列表。
        feature_importance (pd.DataFrame): 特征重要性。
    """
    with open(model_save_path, 'wb') as f:
        pickle.dump({
            'models': models,
            'feature_importance': feature_importance
        }, f)
    logging.info(f"模型和预处理器已保存到 '{model_save_path}'")
