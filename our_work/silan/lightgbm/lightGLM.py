import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import time

# 数据加载和预处理函数
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    if data.columns[0] == 'Unnamed: 0':
        data = data.drop(columns=data.columns[0])
    
    X = data.drop('price', axis=1) if 'price' in data.columns else data
    y = data['price'] if 'price' in data.columns else None
    
    return X, y

# 定义列转换器
def get_column_transformer(X):
    numeric_features = ['curb_weight', 'power', 'engine_cap', 'no_of_owners', 'depreciation', 
                        'coe', 'road_tax', 'dereg_value', 'omv', 'arf', 'vehicle_age']
    # 自动识别其他类别特征
    categorical_features = [col for col in X.columns if col not in numeric_features]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        sparse_threshold=0  # 确保输出为密集矩阵
    )
    
    return preprocessor

# 模型训练和评估函数
def train_evaluate_lightgbm(X_train, X_val, y_train, y_val, params, num_rounds=1000, early_stopping_rounds=50):
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_rounds,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        # early_stopping_rounds=early_stopping_rounds,
        # verbose_eval=100
    )
    
    return model

# 模型评估函数
def evaluate_model(model, X, y):
    predictions = model.predict(X, num_iteration=model.best_iteration)
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    return mse, r2, predictions

# 主函数
def main():
    # 设置随机种子以提高可重复性
    np.random.seed(42)
    
    # 加载和预处理数据
    X, y = load_and_preprocess_data('preprocessing/2024-10-10-silan/train_cleaned.csv')
    
    # 分割数据为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 创建验证集
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    # 预处理数据
    preprocessor = get_column_transformer(X)
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_val_preprocessed = preprocessor.transform(X_val)
    X_test_preprocessed = preprocessor.transform(X_test)
    
    # 保存预处理器
    with open('preprocessor_lightgbm.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    
    # 定义 LightGBM 参数
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': 42,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.2,
        # 'n_jobs': -1, 
    }
    
    # 训练模型
    start_time = time.time()
    model = train_evaluate_lightgbm(X_train_preprocessed, X_val_preprocessed, y_train, y_val, params)
    elapsed_time = time.time() - start_time
    print(f"Training time: {elapsed_time/60:.2f} minutes")
    
    # 评估模型
    mse_train, r2_train, _ = evaluate_model(model, X_train_preprocessed, y_train)
    mse_val, r2_val, _ = evaluate_model(model, X_val_preprocessed, y_val)
    mse_test, r2_test, predictions_test = evaluate_model(model, X_test_preprocessed, y_test)
    
    print(f"Training Set - MSE: {mse_train:.4f}, R2: {r2_train:.4f}")
    print(f"Validation Set - MSE: {mse_val:.4f}, R2: {r2_val:.4f}")
    print(f"Test Set - MSE: {mse_test:.4f}, R2: {r2_test:.4f}")
    
    # 保存模型
    model.save_model('./lightgbm_model.txt')
    print("Model and preprocessor saved")
    
    # 加载测试数据并进行预测
    X_new_test, _ = load_and_preprocess_data('preprocessing/2024-10-10-silan/test_cleaned.csv')
    preprocessor_loaded = pickle.load(open('./preprocessor_lightgbm.pkl', 'rb'))
    X_new_test_preprocessed = preprocessor_loaded.transform(X_new_test)
    
    predictions_new_test = model.predict(X_new_test_preprocessed, num_iteration=model.best_iteration)
    
    # 检查预测结果中的NaN值
    nan_count = np.isnan(predictions_new_test).sum()
    if nan_count > 0:
        print(f"Warning: Found {nan_count} NaN predictions out of {len(predictions_new_test)} total predictions.")
        
        # 找出导致NaN的输入
        nan_indices = np.where(np.isnan(predictions_new_test))[0]
        print("Indices of NaN predictions:", nan_indices)
        
        # 检查这些索引对应的原始输入数据
        problematic_inputs = X_new_test.iloc[nan_indices]
        print("Problematic inputs:")
        print(problematic_inputs)
        
        # 将NaN值替换为训练集的平均值
        train_mean = y_train.mean()
        predictions_new_test[np.isnan(predictions_new_test)] = train_mean
        print(f"Replaced NaN values with training set mean: {train_mean}")
    
    # 创建提交文件，使用round()来处理浮点数
    submission = pd.DataFrame({
        'Id': range(len(predictions_new_test)),
        'Predicted': np.round(predictions_new_test).astype(int)
    })
    
    submission.to_csv('./submission_lightgbm.csv', index=False)
    print("Prediction completed. Submission saved to 'submission_lightgbm.csv'")
    
    # 输出一些统计信息
    print("\nPrediction Statistics:")
    print(f"Min: {predictions_new_test.min()}")
    print(f"Max: {predictions_new_test.max()}")
    print(f"Mean: {predictions_new_test.mean()}")
    print(f"Median: {np.median(predictions_new_test)}")

if __name__ == '__main__':
    main()