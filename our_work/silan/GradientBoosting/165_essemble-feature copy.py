import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
import logging
import joblib
import xgboost as xgb

class WeightedEnsembleL2:
    def __init__(self):
        self.weights = None
        
    def predict(self, predictions):
        """
        使用学习到的权重进行预测
        predictions: shape (n_samples, n_models)
        """
        return predictions @ self.weights

def safe_load_model(model_path):
    """
    Attempts to safely load the model using different methods
    Returns the loaded model or None if loading fails
    """
    try:
        # First attempt: Try regular pickle load
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logging.warning(f"Regular pickle load failed: {str(e)}")
        
        try:
            # Second attempt: Try joblib load
            return joblib.load(model_path)
        except Exception as e:
            logging.warning(f"Joblib load failed: {str(e)}")
            
        try:
            # Third attempt: Try pickle with encoding='latin1'
            with open(model_path, 'rb') as f:
                return pickle.load(f, encoding='latin1')
        except Exception as e:
            logging.warning(f"Pickle load with latin1 encoding failed: {str(e)}")
            
            raise Exception("Failed to load model with all attempted methods. Consider retraining the model with current scikit-learn version.")

def preprocess_features(X, y=None, num_imputer=None, cat_imputer=None, 
                       target_encoder=None, scaler=None, 
                       target_encode_cols=[], 
                       encoding_smoothing=1.0):
    """
    Preprocesses features with imputation, scaling, and encoding
    """
    X = X.copy()
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    columns_to_standardize = ['curb_weight', 'power', 'engine_cap', 'depreciation']
    columns_to_standardize = [col for col in columns_to_standardize if col in X.columns]
    
    if num_imputer is None:
        num_imputer = SimpleImputer(strategy='median')
        X[numeric_features] = pd.DataFrame(num_imputer.fit_transform(X[numeric_features]), 
                                         columns=numeric_features, 
                                         index=X.index)
    else:
        X[numeric_features] = pd.DataFrame(num_imputer.transform(X[numeric_features]), 
                                         columns=numeric_features, 
                                         index=X.index)
    
    if columns_to_standardize:
        if scaler is None:
            scaler = StandardScaler()
            X[columns_to_standardize] = pd.DataFrame(scaler.fit_transform(X[columns_to_standardize]), 
                                                   columns=columns_to_standardize, 
                                                   index=X.index)
        else:
            X[columns_to_standardize] = pd.DataFrame(scaler.transform(X[columns_to_standardize]), 
                                                   columns=columns_to_standardize, 
                                                   index=X.index)

    if len(categorical_features) > 0:
        if cat_imputer is None:
            cat_imputer = SimpleImputer(strategy='constant', fill_value='unknown')
            X[categorical_features] = pd.DataFrame(cat_imputer.fit_transform(X[categorical_features]), 
                                                 columns=categorical_features, 
                                                 index=X.index)
        else:
            X[categorical_features] = pd.DataFrame(cat_imputer.transform(X[categorical_features]), 
                                                 columns=categorical_features, 
                                                 index=X.index)
        
        target_encode_features = [col for col in target_encode_cols if col in categorical_features]
        
        if target_encode_features:
            if target_encoder is None:
                target_encoder = TargetEncoder(cols=target_encode_features, smoothing=encoding_smoothing)
                X[target_encode_features] = pd.DataFrame(target_encoder.fit_transform(X[target_encode_features], y), 
                                                       columns=target_encode_features, 
                                                       index=X.index)
            else:
                X[target_encode_features] = pd.DataFrame(target_encoder.transform(X[target_encode_features]), 
                                                       columns=target_encode_features, 
                                                       index=X.index)
        
        other_categorical = [col for col in categorical_features if col not in target_encode_features]
        if len(other_categorical) > 0:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_features = encoder.fit_transform(X[other_categorical])
            encoded_feature_names = encoder.get_feature_names_out(other_categorical)
            encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=X.index)
            X = pd.concat([X, encoded_df], axis=1)
            X = X.drop(columns=other_categorical)

    return X, num_imputer, cat_imputer, target_encoder, scaler

def predict_with_ensemble(data, model_dict, features_for_clustering=['depreciation', 'coe', 'dereg_value']):
    """
    Makes predictions using an ensemble of models with clustering and WeightedEnsembleL2
    """
    # Unpack model dictionary
    models = model_dict['models']
    kmeans_model = model_dict['kmeans_model']
    cluster_info = model_dict['cluster_info']
    
    # Prepare data for clustering
    if 'price' in data.columns:
        data = data.drop(columns=['price'])
    
    # Standardize numerical features
    data = data.copy()
    for col in ['curb_weight', 'power', 'engine_cap']:
        if col in data.columns:
            data[col] = (data[col] - data[col].mean()) / data[col].std()
    
    # Create features for clustering
    dummy_y = np.zeros(len(data))
    cluster_features = np.column_stack([
        np.log1p(dummy_y), 
        data[features_for_clustering]
    ])
    
    # Predict clusters
    clusters = kmeans_model.predict(cluster_features)
    
    # Initialize predictions
    final_predictions = np.zeros(len(data))
    
    # Make predictions for each cluster
    for cluster in range(len(cluster_info)):
        cluster_mask = clusters == cluster
        X_cluster = data[cluster_mask]
        
        if len(X_cluster) == 0:
            continue
            
        cluster_predictions = np.zeros(len(X_cluster))
        num_models = 0
        
        # Use all models for current cluster
        for model_dict in models[cluster]:
            model_lgb = model_dict['lightgbm']
            model_xgb = model_dict['xgboost']
            model_gb = model_dict['gradient_boosting']
            model_cb = model_dict['catboost']
            ensemble_weights = model_dict['ensemble_weights']
            preprocessors = model_dict['preprocessors']
            
            try:
                # Preprocess data
                X_processed, _, _, _, _ = preprocess_features(
                    X_cluster, 
                    y=None,
                    num_imputer=preprocessors['num_imputer'],
                    cat_imputer=preprocessors['cat_imputer'],
                    target_encoder=preprocessors['target_encoder'],
                    scaler=preprocessors['scaler']
                )
                
                # Make predictions with all four models
                pred_lgb = np.expm1(model_lgb.predict(X_processed))
                pred_xgb = np.expm1(model_xgb.predict(xgb.DMatrix(X_processed)))
                pred_gb = np.expm1(model_gb.predict(X_processed))
                pred_cb = np.expm1(model_cb.predict(X_processed))
                
                # Stack predictions and apply ensemble weights
                predictions_stack = np.column_stack([pred_lgb, pred_xgb, pred_gb, pred_cb])
                weighted_predictions = predictions_stack @ ensemble_weights
                
                cluster_predictions += weighted_predictions
                num_models += 1
                
            except Exception as e:
                logging.error(f"Error predicting for cluster {cluster} with one of the models: {str(e)}")
                continue
            
        if num_models > 0:
            final_predictions[cluster_mask] = cluster_predictions / num_models
    
    return final_predictions

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Load model with safe loader
        model_path = 'ensemble_clustered_models_new.pkl'
        logging.info("Attempting to load model...")
        model_dict = safe_load_model(model_path)
        
        if model_dict is None:
            raise Exception("Model loading failed with all attempts")
            
        # Load and prepare data
        train_data = pd.read_csv('preprocessing//release//ver2//train_cleaned.csv')
        test_data = pd.read_csv('preprocessing//release//ver2//test_cleaned.csv')
        print('Initial shapes:')
        print('train_data.shape', train_data.shape)
        print('test_data.shape', test_data.shape)

        # Remove first column
        train_data = train_data.drop(columns=train_data.columns[0])
        test_data = test_data.drop(columns=test_data.columns[0])

        # Make predictions
        logging.info("Making predictions for train data...")
        train_predictions = predict_with_ensemble(train_data, model_dict)
        logging.info("Making predictions for test data...")
        test_predictions = predict_with_ensemble(test_data, model_dict)

        # Post-process predictions
        min_price = 700
        max_price = 2900000
        train_predictions = np.clip(train_predictions, min_price, max_price)
        test_predictions = np.clip(test_predictions, min_price, max_price)

        # Create new datasets with predictions
        l2_train = train_data.copy()
        l2_test = test_data.copy()
        
        l2_train['ref_price'] = train_predictions
        l2_test['ref_price'] = test_predictions
        
        # Save datasets
        l2_train.to_csv('l2_train_2.csv', index=False)
        l2_test.to_csv('l2_test_2.csv', index=False)
        
        # Create submission file
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
        
        print("\nPrediction statistics:")
        print(f"Train predictions - Min: {train_predictions.min():.2f}, Max: {train_predictions.max():.2f}")
        print(f"Test predictions - Min: {test_predictions.min():.2f}, Max: {test_predictions.max():.2f}")
        
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()