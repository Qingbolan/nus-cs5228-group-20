import pickle
import time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# 设定模型参数
model_GBoost = GradientBoostingRegressor(
    n_estimators=3000,  # 增加迭代次数，提升模型学习能力
    learning_rate=0.1,  # 降低学习率，结合增加的迭代次数来增强泛化能力
    max_depth=5,  # 增加树的深度，以更好地捕捉复杂的模式
    max_features='sqrt',
    min_samples_leaf=15,  # 减少叶子节点最小样本数以捕获更多信息
    min_samples_split=10,  # 减少分裂的最小样本数
    loss='huber',
    random_state=42
)


# 定义RMSE计算函数
def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# 训练和评估函数
def train_and_evaluate(train_idx, val_idx, X, y, model):
    X_fold_train, X_fold_val = X[train_idx], X[val_idx]
    y_fold_train, y_fold_val = y[train_idx], y[val_idx]

    model.fit(X_fold_train, y_fold_train)
    y_fold_pred = model.predict(X_fold_val)

    fold_rmse = rmse(y_fold_val, y_fold_pred)
    return fold_rmse


# K-fold Cross Validation with parallel processing
def rmsle_cross_val(X_train, y_train, n_folds, model, n_jobs=-1):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    rmse_scores = Parallel(n_jobs=n_jobs)(
        delayed(train_and_evaluate)(train_idx, val_idx, X_train.values, y_train.values, model)
        for train_idx, val_idx in kf.split(X_train)
    )

    for fold_idx, fold_rmse in enumerate(rmse_scores):
        print(f"Fold {fold_idx + 1}: RMSE = {fold_rmse:.4f}")

    return np.array(rmse_scores)


# 主函数
if __name__ == '__main__':
    # 训练集数据加载
    train_data = pd.read_csv('/Users/williampang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/24fall---semester1/CS5228-Knowledge Discovery and Data Mining/Project/Weicong-Process/Data-Processed/train_cleaned.csv')

    # 删除第0列（索引列）
    if 'Unnamed: 0' in train_data.columns:
        train_data = train_data.drop(columns='Unnamed: 0')

    # 提取目标变量 'price'
    train_labels = train_data['price']
    train_data = train_data.drop(columns=['price'])

    # # 对 'make' 和 'model' 进行标签编码
    # le_make = LabelEncoder()
    # le_model = LabelEncoder()
    # train_data['make'] = le_make.fit_transform(train_data['make'])
    # train_data['model'] = le_model.fit_transform(train_data['model'])

    # 删除make和model两个列
    train_data = train_data.drop(columns=['make', 'model'])



    # 数值型特征的标准化
    numerical_columns = ['engine_cap', 'power', 'curb_weight']
    scaler = StandardScaler()
    train_data[numerical_columns] = scaler.fit_transform(train_data[numerical_columns])

    # 输出数据集的形状
    print('train_data.shape', train_data.shape)

    # 开始计时
    start_time = time.time()

    # 使用K折交叉验证评价模型性能
    score_GBoost = rmsle_cross_val(train_data, np.log1p(train_labels), 10, model_GBoost)
    print(f"GradientBoostingRegressor average score: {score_GBoost.mean():.4f} ({score_GBoost.std():.4f})\n")

    # 结束计时，计算训练时间
    elapsed_time = time.time() - start_time
    print(f"Time taken for training: {elapsed_time:.2f} seconds")

    # 最终在整个数据集上训练模型
    model_GBoost.fit(train_data, train_labels)

    # 保存模型
    model_path = 'gboost_model_optimized.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_GBoost, f)

    print(f"Training Completed. Model Saved at {model_path}")

    # 加载训练好的模型
    with open(model_path, 'rb') as f:
        model_GBoost = pickle.load(f)

    # 读取测试集数据
    test_data = pd.read_csv('/Users/williampang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/24fall---semester1/CS5228-Knowledge Discovery and Data Mining/Project/Weicong-Process/Data-Processed/test_cleaned.csv')

    # 删除第0列（假设需要删除，和train类似）
    if 'Unnamed: 0' in test_data.columns:
        test_data = test_data.drop(columns='Unnamed: 0')
    #
    # # 对 'make' 和 'model' 进行标签编码
    # test_data['make'] = le_make.transform(test_data['make'])
    # test_data['model'] = le_model.transform(test_data['model'])

    # 删除make和model两个列
    test_data = test_data.drop(columns=['make', 'model'])


    # 数值特征标准化，需要使用训练集的scaler
    test_data[numerical_columns] = scaler.transform(test_data[numerical_columns])

    print('test_data.shape', test_data.shape)

    # 使用numpy array进行预测
    predictions = model_GBoost.predict(test_data)

    # 识别没有空值的行和有空值的行
    no_nan_rows = test_data.dropna()  # 没有空值的行
    nan_rows = test_data[test_data.isna().any(axis=1)]  # 有空值的行

    # 对没有空值的行进行预测
    predictions_no_nan = model_GBoost.predict(no_nan_rows)

    # 计算预测结果的均值
    mean_prediction = predictions_no_nan.mean()

    # 创建一个与测试集大小相同的空预测数组
    predictions = np.empty(test_data.shape[0])

    # 对没有空值的行进行预测
    predictions[no_nan_rows.index] = predictions_no_nan

    # 对有空值的行填充预测均值
    predictions[nan_rows.index] = mean_prediction

    # 创建提交文件的格式
    submission = pd.DataFrame({
        'Id': range(len(predictions)),
        'Predicted': predictions.astype(int)  # 直接转换为整数
    })

    # 保存提交文件，确保格式符合要求
    submission.to_csv('submission_version_two_group20.csv', index=False)

    print("Prediction Completed. Submission saved to 'submission_version_two_group20.csv'")
