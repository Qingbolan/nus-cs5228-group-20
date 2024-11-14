# binary_classifier.py

import os
import pickle
import logging
import json
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, auc,
                             precision_recall_curve, average_precision_score)
from sklearn.model_selection import (train_test_split, GridSearchCV, cross_val_score, 
                                     learning_curve, validation_curve)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.calibration import calibration_curve

class MistakeClassifier:
    """集成了训练、评估和可视化的分类器类"""
    
    def __init__(
            self,
            output_dir: str = 'model_evaluation',
            classifier_name: str = 'xgboost',
            random_state: int = 42
    ):
        self.output_dir = output_dir
        self.classifier_name = classifier_name
        self.random_state = random_state
        self.evaluation_results = {}
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)
        
        # 配置日志
        log_file = os.path.join(output_dir, 'training.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _get_classifier(self, params: Dict = None) -> Any:
        """根据名称和参数获取分类器实例"""
        if self.classifier_name == 'xgboost':
            default_params = {
                'max_depth': 6,
                'n_estimators': 100,
                'learning_rate': 0.1,
                'random_state': self.random_state
            }
        elif self.classifier_name == 'random_forest':
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'random_state': self.random_state
            }
        elif self.classifier_name == 'logistic_regression':
            default_params = {
                'multi_class': 'multinomial',
                'solver': 'lbfgs',
                'max_iter': 1000,
                'random_state': self.random_state
            }
        else:
            raise ValueError(f"不支持的分类器: {self.classifier_name}")
            
        final_params = {**default_params, **(params or {})}
        
        if self.classifier_name == 'xgboost':
            return xgb.XGBClassifier(**final_params)
        elif self.classifier_name == 'random_forest':
            return RandomForestClassifier(**final_params)
        else:
            return LogisticRegression(**final_params)
    
    def _generate_visualizations(
            self,
            classifier: Any,
            X_train: pd.DataFrame,
            X_test: pd.DataFrame,
            y_train: np.ndarray,
            y_test: np.ndarray,
            feature_names: List[str],
            class_names: List[Any]
    ):
        """生成所有可视化图表"""
        plots_dir = os.path.join(self.output_dir, 'plots')
        
        # 1. 特征重要性
        if hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_
        elif hasattr(classifier, "coef_"):
            importances = np.mean(np.abs(classifier.coef_), axis=0)
        else:
            importances = None
            
        if importances is not None:
            plt.figure(figsize=(12, 6))
            indices = np.argsort(importances)[::-1][:20]  # Top 20 features
            plt.title('Feature Importance')
            plt.bar(range(len(indices)), importances[indices])
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'feature_importance.png'))
            plt.close()
            logging.info(f"特征重要性图表已保存在: {os.path.join(plots_dir, 'feature_importance.png')}")
            
            # 保存特征重要性数据
            feature_importance_data = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            feature_importance_data.to_csv(
                os.path.join(self.output_dir, 'reports', 'feature_importance.csv'),
                index=False
            )
        
        # 2. 混淆矩阵
        cm = confusion_matrix(y_test, classifier.predict(X_test))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'))
        logging.info(f"混淆矩阵已保存在: {os.path.join(plots_dir, 'confusion_matrix.png')}")
        plt.close()
        
        # 3. 学习曲线
        train_sizes, train_scores, test_scores = learning_curve(
            classifier, X_train, y_train, cv=5,
            n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
        plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
        plt.xlabel('Training examples')
        plt.ylabel('Score')
        plt.title('Learning Curves')
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(os.path.join(plots_dir, 'learning_curves.png'))
        logging.info(f"学习曲线已保存在: {os.path.join(plots_dir, 'learning_curves.png')}")
        plt.close()
        
        # 4. ROC曲线 (二分类情况)
        if len(class_names) == 2 and hasattr(classifier, "predict_proba"):
            y_pred_proba = classifier.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                     label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(plots_dir, 'roc_curve.png'))
            plt.close()
            logging.info(f"ROC曲线已保存在: {os.path.join(plots_dir, 'roc_curve.png')}")
            
            self.evaluation_results['roc_auc'] = roc_auc
    
    def _generate_evaluation_report(
            self,
            classifier: Any,
            X_train: pd.DataFrame,
            X_test: pd.DataFrame,
            y_train: np.ndarray,
            y_test: np.ndarray,
            class_names: List[Any]  # 将类型改为 Any，适应可能的数值类型
    ) -> str:
        """生成详细的评估报告"""
        y_pred = classifier.predict(X_test)
        
        # 转换 class_names 为字符串列表
        class_names = [str(cn) for cn in class_names]
        
        # 基础分类报告
        class_report = classification_report(y_test, y_pred, 
                                             target_names=class_names, 
                                             output_dict=True,
                                             zero_division=0)
        
        # 交叉验证分数
        cv_scores = cross_val_score(classifier, X_train, y_train, cv=5)
        
        # 汇总评估结果
        self.evaluation_results.update({
            'classifier_name': self.classifier_name,
            'classification_report': class_report,
            'cv_scores': {
                'mean': float(cv_scores.mean()),
                'std': float(cv_scores.std()),
                'scores': cv_scores.tolist()
            },
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        # 生成文本报告
        report_lines = [
            "=== 分类器评估报告 ===",
            f"分类器: {self.classifier_name}",
            f"评估时间: {self.evaluation_results['training_date']}",
            "\n=== 分类报告 ===",
            classification_report(y_test, y_pred, target_names=class_names, zero_division=0),
            "\n=== 交叉验证结果 ===",
            f"平均分数: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})",
        ]
        
        # 如果是二分类问题且分类器支持概率预测
        if len(class_names) == 2 and hasattr(classifier, "predict_proba"):
            y_pred_proba = classifier.predict_proba(X_test)[:, 1]
            report_lines.extend([
                f"\n=== ROC AUC 分数 ===",
                f"ROC AUC: {self.evaluation_results['roc_auc']:.3f}"
            ])
        
        # 保存详细结果到JSON
        with open(os.path.join(self.output_dir, 'reports', 'evaluation_results.json'), 'w') as f:
            json.dump(self.evaluation_results, f, indent=4)
        
        # 保存文本报告
        report_text = '\n'.join(report_lines)
        with open(os.path.join(self.output_dir, 'reports', 'evaluation_report.txt'), 'w') as f:
            f.write(report_text)
        
        return report_text
    
    def train_and_evaluate(
            self,
            data: pd.DataFrame,
            test_size: float = 0.2,
            use_grid_search: bool = True
    ) -> Dict[str, Any]:
        """训练分类器并生成完整的评估报告"""
        logging.info(f"开始训练 {self.classifier_name} 分类器")
        
        # 数据准备
        X = data.drop(columns=['price', 'inference_price', 'mistake', 'mistake_abs', 
                               'mistake_category', 'Id'], errors='ignore')
        y = data['mistake_category']
        
        # 标签编码
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # 数据集划分
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=test_size, 
            random_state=self.random_state, 
            stratify=y_encoded
        )
        
        # 网格搜索（如果启用）
        if use_grid_search:
            param_grid = {
                'xgboost': {
                    'max_depth': [3, 5, 7],
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1]
                },
                'random_forest': {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                },
                'logistic_regression': {
                    'C': [0.1, 1.0, 10.0],
                    'solver': ['lbfgs', 'newton-cg']
                }
            }.get(self.classifier_name, {})
            
            if param_grid:
                grid_search = GridSearchCV(
                    self._get_classifier(),
                    param_grid,
                    cv=5,
                    scoring='f1_weighted',
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                classifier = self._get_classifier(grid_search.best_params_)
                self.evaluation_results['best_params'] = grid_search.best_params_
            else:
                classifier = self._get_classifier()
        else:
            classifier = self._get_classifier()
        
        # 训练最终模型
        classifier.fit(X_train, y_train)
        
        # 生成可视化
        self._generate_visualizations(
            classifier,
            X_train,
            X_test,
            y_train,
            y_test,
            X.columns.tolist(),
            label_encoder.classes_
        )
        
        # 生成评估报告
        evaluation_report = self._generate_evaluation_report(
            classifier,
            X_train,
            X_test,
            y_train,
            y_test,
            label_encoder.classes_
        )
        
        logging.info("评估报告已生成")
        logging.info("\n" + evaluation_report)
        
        # 保存模型
        model_save_path = os.path.join(self.output_dir, 'model.pkl')
        with open(model_save_path, 'wb') as f:
            pickle.dump({
                'classifier': classifier,
                'label_encoder': label_encoder,
                'feature_names': X.columns.tolist(),
                'evaluation_results': self.evaluation_results,
                'scaler': None,  # 如果有预处理器，可以在这里添加
                'num_imputer': None
            }, f)
        
        logging.info(f"模型已保存到: {model_save_path}")
        
        return {
            'classifier': classifier,
            'label_encoder': label_encoder,
            'evaluation_results': self.evaluation_results,
            'model_path': model_save_path,
            'scaler': None,
            'num_imputer': None
        }

def train_and_evaluate_classifier(
        data: pd.DataFrame,
        classifier_name: str = 'xgboost',
        output_dir: str = 'model_evaluation',
        test_size: float = 0.2,
        use_grid_search: bool = True,
        random_state: int = 42
) -> Dict[str, Any]:
    """便捷函数：训练和评估分类器

    Args:
        data: 包含特征和目标变量的数据框
        classifier_name: 分类器名称 ('xgboost', 'random_forest', 或 'logistic_regression')
        output_dir: 输出目录路径
        test_size: 测试集比例
        use_grid_search: 是否使用网格搜索优化参数
        random_state: 随机种子

    Returns:
        Dict: 包含训练好的分类器和评估结果的字典
    """
    classifier = MistakeClassifier(
        output_dir=output_dir,
        classifier_name=classifier_name,
        random_state=random_state
    )
    
    results = classifier.train_and_evaluate(
        data=data,
        test_size=test_size,
        use_grid_search=use_grid_search
    )
    
    return results

def predict_mistake_category(
        data: pd.DataFrame,
        model_path: str
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """使用保存的分类器模型进行误差类别预测

    Args:
        data: 需要预测的数据（DataFrame）
        model_path: 保存的分类器模型文件路径

    Returns:
        Tuple: 包含预测的类别和预测信息的元组
    """
    # 加载模型
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"分类器模型文件未找到: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    classifier = model_data['classifier']
    label_encoder = model_data['label_encoder']
    feature_names = model_data['feature_names']
    # 如果有预处理器，可以在这里加载
    scaler = model_data.get('scaler', None)
    num_imputer = model_data.get('num_imputer', None)
    
    # 准备特征
    X = data[feature_names].copy()
    
    # 预处理（如果有预处理器）
    if num_imputer:
        X = num_imputer.transform(X)
    if scaler:
        X = scaler.transform(X)
    
    # 预测
    predictions_encoded = classifier.predict(X)
    predicted_categories = label_encoder.inverse_transform(predictions_encoded)
    
    # 返回预测结果和信息
    prediction_info = {
        'predicted_categories': predicted_categories,
        'class_names': label_encoder.classes_
    }
    
    return predicted_categories, prediction_info