import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 加载训练好的模型
model_path = 'gboost_model.pkl'
with open(model_path, 'rb') as f:
    model_GBoost = pickle.load(f)

# 读取训练集和测试集数据
train_data = pd.read_csv('preprocessing/release/ver2/train_cleaned.csv')
test_data = pd.read_csv('preprocessing/release/ver2/test_cleaned.csv')
print('train_data.shape', train_data.shape)
print('test_data.shape', test_data.shape)

# 删除第0列
train_data = train_data.drop(columns=train_data.columns[0])
test_data = test_data.drop(columns=test_data.columns[0])

# 保存原始price列
original_price = None
if 'price' in train_data.columns:
    original_price = train_data['price'].copy()
    train_data = train_data.drop(columns=['price'])

# 对训练集数据标准化
train_data['curb_weight'] = (train_data['curb_weight'] - train_data['curb_weight'].mean()) / train_data['curb_weight'].std()
train_data['power'] = (train_data['power'] - train_data['power'].mean()) / train_data['power'].std()
train_data['engine_cap'] = (train_data['engine_cap'] - train_data['engine_cap'].mean()) / train_data['engine_cap'].std()

# 对测试集数据标准化
test_data['curb_weight'] = (test_data['curb_weight'] - test_data['curb_weight'].mean()) / test_data['curb_weight'].std()
test_data['power'] = (test_data['power'] - test_data['power'].mean()) / test_data['power'].std()
test_data['engine_cap'] = (test_data['engine_cap'] - test_data['engine_cap'].mean()) / test_data['engine_cap'].std()

# 打印特征数量检查
print('\nFeature count check:')
print('Train features:', train_data.shape[1])
print('Test features:', test_data.shape[1])


# 训练集预测
train_no_nan = train_data.dropna()
train_nan = train_data[train_data.isna().any(axis=1)]

train_predictions_no_nan = model_GBoost.predict(train_no_nan.values)
train_mean_prediction = train_predictions_no_nan.mean()

train_predictions = np.empty(train_data.shape[0])
train_predictions[train_no_nan.index] = train_predictions_no_nan
train_predictions[train_nan.index] = train_mean_prediction

# 测试集预测
test_no_nan = test_data.dropna()
test_nan = test_data[test_data.isna().any(axis=1)]

test_predictions_no_nan = model_GBoost.predict(test_no_nan.values)
test_mean_prediction = test_predictions_no_nan.mean()

test_predictions = np.empty(test_data.shape[0])
test_predictions[test_no_nan.index] = test_predictions_no_nan
test_predictions[test_nan.index] = test_mean_prediction

# 创建包含预测结果的新数据集
l2_train = train_data.copy()
l2_test = test_data.copy()

# 添加原始price列回到训练集
if original_price is not None:
    l2_train['price'] = original_price

# 添加预测价格作为新列
l2_train['ref_price'] = train_predictions
l2_test['ref_price'] = test_predictions

# 验证列的存在
print('\nColumn verification:')
print('Train columns:', l2_train.columns.tolist())
print('Test columns:', l2_test.columns.tolist())

# 保存新的数据集
l2_train.to_csv('l2_train.csv', index=False)
l2_test.to_csv('l2_test.csv', index=False)

# 创建提交文件
formatted_predictions = [f'{int(pred)}.0' for pred in test_predictions]
submission = pd.DataFrame({
    'Id': range(len(test_predictions)),
    'Predicted': formatted_predictions
})
submission.to_csv('submission_version_one_group20.csv', index=False)

print("\nProcessing completed successfully!")
print("1. L2 training data saved to 'l2_train.csv'")
print("2. L2 test data saved to 'l2_test.csv'")
print("3. Submission file saved to 'submission_version_one_group20.csv'")

# 最终验证
final_train = pd.read_csv('l2_train.csv')
final_test = pd.read_csv('l2_test.csv')
print('\nFinal verification:')
print('Train file columns:', final_train.columns.tolist())
print('Test file columns:', final_test.columns.tolist())