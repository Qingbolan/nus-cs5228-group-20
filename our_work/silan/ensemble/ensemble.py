import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from catboost import CatBoostRegressor

# 函数：加载模型
def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# 函数：数据预处理
def preprocess_data(data):
    # 删除第0列
    data = data.drop(columns=data.columns[0])
    
    # 对 'make' 和 'model' 列进行标签编码
    le_make = LabelEncoder()
    le_model = LabelEncoder()
    data['make'] = le_make.fit_transform(data['make'])
    data['model'] = le_model.fit_transform(data['model'])
    
    # 对指定列进行标准化
    for col in ['curb_weight', 'power', 'engine_cap']:
        data[col] = (data[col] - data[col].mean()) / data[col].std()
    
    return data

# 函数：进行预测
def predict(model, data):
    # 识别没有空值的行和有空值的行
    no_nan_rows = data.dropna()
    nan_rows = data[data.isna().any(axis=1)]
    
    # 对没有空值的行进行预测
    predictions_no_nan = model.predict(no_nan_rows.values)
    
    # 计算预测结果的均值
    mean_prediction = predictions_no_nan.mean()
    
    # 创建一个与数据集大小相同的空预测数组
    predictions = np.empty(data.shape[0])
    
    # 填充预测结果
    predictions[no_nan_rows.index] = predictions_no_nan
    predictions[nan_rows.index] = mean_prediction
    
    return predictions

# 主程序
if __name__ == "__main__":
    # 读取测试集数据
    test_data = pd.read_csv('preprocessing/2024-10-10-silan/test_cleaned.csv')
    print('Original test_data.shape', test_data.shape)
    
    # 预处理数据
    test_data = preprocess_data(test_data)
    print('Preprocessed test_data.shape', test_data.shape)
    
    # 加载模型
    model_GBoost = load_model('gboost_model.pkl')
    model_CatBoost = load_model('catboost_model.pkl')
    model_LightGBM = lgb.Booster(model_file='lightgbm_model_selective_standardized.txt')
    
    # 进行预测
    pred_GBoost = predict(model_GBoost, test_data)
    pred_CatBoost = predict(model_CatBoost, test_data)
    pred_LightGBM = model_LightGBM.predict(test_data)
    
    # 定义模型权重（请根据实际情况调整）
    weights = {
        'GBoost': 0.4,
        'CatBoost': 0.3,
        'LightGBM': 0.3
    }
    
    # 计算加权平均预测
    ensemble_pred = (
        weights['GBoost'] * pred_GBoost +
        weights['CatBoost'] * pred_CatBoost +
        weights['LightGBM'] * pred_LightGBM
    )
    
    # 将预测结果转换为整数并格式化为以 .0 结尾
    formatted_predictions = [f'{int(pred)}.0' for pred in ensemble_pred]
    
    # 创建提交文件
    submission = pd.DataFrame({
        'Id': range(len(ensemble_pred)),
        'Predicted': formatted_predictions
    })
    
    # 保存提交文件
    submission.to_csv('submission_ensemble_group20.csv', index=False)
    
    print("Ensemble Prediction Completed. Submission saved to 'submission_ensemble_group20.csv'")