import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 加载训练好的模型
model_path = 'catboost_model.pkl'
with open(model_path, 'rb') as f:
    model_GBoost = pickle.load(f)

# 读取测试集数据
test_data = pd.read_csv('preprocessing/2024-10-10-silan/test_cleaned.csv')

print('test_data.shape', test_data.shape)

# 删除第0列（假设需要删除，和train类似）
test_data = test_data.drop(columns=test_data.columns[0])

# 创建 LabelEncoder 对象
le_make = LabelEncoder()
le_model = LabelEncoder()

# # 对 'make' 列进行标签编码
# test_data['make'] = le_make.fit_transform(test_data['make'])

# # 对 'model' 列进行标签编码
# test_data['model'] = le_model.fit_transform(test_data['model'])

# 对 ['curb_weight', 'power', 'engine_cap','depreciation'] 列分别进行标准化
test_data['curb_weight'] = (test_data['curb_weight'] - test_data['curb_weight'].mean()) / test_data['curb_weight'].std()
test_data['power'] = (test_data['power'] - test_data['power'].mean()) / test_data['power'].std()
test_data['engine_cap'] = (test_data['engine_cap'] - test_data['engine_cap'].mean()) / test_data['engine_cap'].std()

# 确保测试数据的列与训练数据一致
print('test_data.shape', test_data.shape)

# 识别没有空值的行和有空值的行
no_nan_rows = test_data.dropna()  # 没有空值的行
nan_rows = test_data[test_data.isna().any(axis=1)]  # 有空值的行

# 对没有空值的行进行预测
predictions_no_nan = model_GBoost.predict(no_nan_rows.values)

# 计算预测结果的均值
mean_prediction = predictions_no_nan.mean()

# 创建一个与测试集大小相同的空预测数组
predictions = np.empty(test_data.shape[0])

# 对没有空值的行进行预测
predictions[no_nan_rows.index] = predictions_no_nan

# 对有空值的行填充预测均值
predictions[nan_rows.index] = mean_prediction

# 将预测结果转换为整数并格式化为以 .0 结尾
formatted_predictions = [f'{int(pred)}.0' for pred in predictions]

# 创建提交文件的格式
submission = pd.DataFrame({
    'Id': range(len(predictions)),  # 创建从0开始的索引
    'Predicted': formatted_predictions
})

# 保存提交文件，确保格式符合要求
submission.to_csv('submission_catboost.csv', index=False)

print("Prediction Completed. Submission saved to 'submission_catboost.csv'")