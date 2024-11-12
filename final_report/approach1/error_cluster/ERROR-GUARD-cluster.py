# binary_classification.py

def main():
    # import argparse
    
    # parser = argparse.ArgumentParser(description='训练二分类模型')
    # parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    # parser.add_argument('--base_model', type=str, required=True, 
    #                   choices=['lightgbm', 'xgboost', 'catboost', 'gradient_boosting'],
    #                   help='原始预测模型的名称')
    
    # args = parser.parse_args()
    
    config = 'config.yaml'
    base_model = 'lightgbm'
    
    
    # 初始化训练器
    trainer = BinaryClassificationTrainer(config)
    
    # 准备数据
    X, y = trainer.prepare_data(base_model)
    
    # 分割数据
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=trainer.config['common']['random_state'],
        stratify=y
    )
    
    # 训练模型
    model_info = trainer.train_model(X_train, y_train, X_val, y_val, base_model)
    
    logging.info(f"二分类模型训练完成 (基于 {base_model} 的预测结果)")



import os
import logging
import yaml
import pickle
from pathlib import Path
from typing import Dict, Union, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

class BinaryClassificationTrainer:
    """
    基于XGBoost的二分类模型训练器
    """
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.setup_directories()
        self.setup_logging()
    
    def setup_directories(self):
        """创建必要的目录"""
        os.makedirs(self.config['common']['output_dir'], exist_ok=True)
        os.makedirs(self.config['binary_classifier']['save_dir'], exist_ok=True)
    
    def setup_logging(self):
        """配置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('binary_classification.log'),
                logging.StreamHandler()
            ]
        )
    
    def prepare_data(self, base_model_name: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备训练数据
        
        Args:
            base_model_name: 原始预测模型的名称（如'lightgbm', 'xgboost'等）
        """
        # 获取对应模型的推理结果路径
        results_path = self.config['data']['base_models_results'][base_model_name]
        
        # 读取推理结果
        df = pd.read_csv(results_path)
        
        # 计算误差并创建标签
        df['mistake'] = abs(df['price'] - df['inference_price'])
        binary_labels = (df['mistake'] > self.config['common']['error_threshold']).astype(int)
        
        # 选择特征
        features = df[self.config['data']['feature_columns']]
        
        # 打印类别分布
        class_dist = binary_labels.value_counts(normalize=True)
        logging.info(f"Class distribution for {base_model_name}:\n{class_dist}")
        
        return features, binary_labels
    
    def plot_feature_importance(self, model: xgb.XGBClassifier, feature_names: List[str]):
        """绘制特征重要性图"""
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=importance_df.head(20), x='importance', y='feature')
        plt.title('Top 20 Important Features')
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        
        save_path = os.path.join(self.config['common']['output_dir'], 'feature_importance.png')
        plt.savefig(save_path)
        plt.close()
        
        return importance_df
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, base_model_name: str):
        """绘制ROC曲线"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {base_model_name}')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        save_path = os.path.join(self.config['common']['output_dir'], f'roc_curve_{base_model_name}.png')
        plt.savefig(save_path)
        plt.close()
    
    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        base_model_name: str
    ) -> Dict:
        """
        训练XGBoost二分类模型
        
        Args:
            base_model_name: 用于标识是哪个原始预测模型的二分类器
        """
        # 计算正负样本比例，更新scale_pos_weight
        pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()
        self.config['binary_classifier']['params']['scale_pos_weight'] = pos_ratio
        
        # 创建模型
        model = xgb.XGBClassifier(
            **self.config['binary_classifier']['params'],
            random_state=self.config['common']['random_state']
        )
        
        # 训练模型
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            early_stopping_rounds=self.config['binary_classifier']['training']['early_stopping_rounds'],
            verbose=self.config['binary_classifier']['training']['verbose_eval']
        )
        
        # 评估模型
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # 生成评估报告
        evaluation = {
            'classification_report': classification_report(y_val, y_pred),
            'confusion_matrix': confusion_matrix(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_pred_proba),
            'average_precision': average_precision_score(y_val, y_pred_proba)
        }
        
        # 保存评估报告
        report_path = os.path.join(
            self.config['common']['output_dir'],
            f'evaluation_{base_model_name}.txt'
        )
        with open(report_path, 'w') as f:
            f.write(f"Model Evaluation for {base_model_name}\n\n")
            f.write(f"Classification Report:\n{evaluation['classification_report']}\n\n")
            f.write(f"Confusion Matrix:\n{evaluation['confusion_matrix']}\n\n")
            f.write(f"ROC AUC Score: {evaluation['roc_auc']:.4f}\n")
            f.write(f"Average Precision Score: {evaluation['average_precision']:.4f}\n")
        
        # 绘制ROC曲线
        self.plot_roc_curve(y_val, y_pred_proba, base_model_name)
        
        # 分析特征重要性
        importance_df = self.plot_feature_importance(model, list(X_train.columns))
        
        # 保存模型和相关信息
        model_info = {
            'model': model,
            'feature_columns': list(X_train.columns),
            'error_threshold': self.config['common']['error_threshold'],
            'evaluation': evaluation,
            'feature_importance': importance_df
        }
        
        save_path = os.path.join(
            self.config['binary_classifier']['save_dir'],
            f'binary_classifier_{base_model_name}.pkl'
        )
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_info, f)
        
        return model_info

def predict_error_category(
    model_path: str,
    data: pd.DataFrame
) -> np.ndarray:
    """
    使用训练好的二分类模型预测误差类别
    
    Args:
        model_path: 模型文件路径
        data: 包含特征的DataFrame
    
    Returns:
        预测的二分类结果: 0表示误差较小，1表示误差较大
    """
    # 加载模型
    with open(model_path, 'rb') as f:
        model_info = pickle.load(f)
    
    # 确保数据包含所需特征
    required_features = model_info['feature_columns']
    if not all(feat in data.columns for feat in required_features):
        raise ValueError("数据缺少必要的特征")
    
    # 进行预测
    X = data[required_features]
    model = model_info['model']
    predictions = model.predict(X)
    
    return predictions

if __name__ == '__main__':
    main()