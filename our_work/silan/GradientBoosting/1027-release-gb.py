import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
import pickle
import time
from category_encoders import TargetEncoder
import logging
import warnings
from typing import Tuple, List, Dict, Any, Optional
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)

def load_and_preprocess_data(file_path: str) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Load and perform initial preprocessing of data.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        Tuple[pd.DataFrame, Optional[pd.Series]]: Features (X) and target (y) if available
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Successfully loaded data from {file_path}. Shape: {data.shape}")
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {str(e)}")
        raise
    
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns='Unnamed: 0')
    
    X = data.drop('price', axis=1) if 'price' in data.columns else data
    y = data['price'] if 'price' in data.columns else None
    
    logging.info(f"Features shape: {X.shape}")
    if y is not None:
        logging.info(f"Target shape: {y.shape}")
        logging.info(f"Price range: {y.min():.2f} to {y.max():.2f}")
    
    return X, y

def preprocess_features(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    num_imputer: Optional[SimpleImputer] = None,
    cat_imputer: Optional[SimpleImputer] = None,
    target_encoder: Optional[TargetEncoder] = None,
    scaler: Optional[StandardScaler] = None,
    target_encode_cols: List[str] = ['make', 'model'],
    encoding_smoothing: float = 1.0
) -> Tuple[pd.DataFrame, SimpleImputer, SimpleImputer, TargetEncoder, StandardScaler]:
    """
    Preprocess features including imputation, scaling, and encoding.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    
    X = X.copy()
    logging.info("Starting feature preprocessing")
    
    # Identify feature types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    logging.info(f"Numeric features: {numeric_features.tolist()}")
    logging.info(f"Categorical features: {categorical_features.tolist()}")
    
    # Standard columns to scale
    columns_to_standardize = ['curb_weight', 'power', 'engine_cap', 'depreciation']
    columns_to_standardize = [col for col in columns_to_standardize if col in X.columns]
    
    # Handle numeric features
    if len(numeric_features) > 0:
        if num_imputer is None:
            num_imputer = SimpleImputer(strategy='median')
            X[numeric_features] = pd.DataFrame(
                num_imputer.fit_transform(X[numeric_features]),
                columns=numeric_features,
                index=X.index
            )
        else:
            X[numeric_features] = pd.DataFrame(
                num_imputer.transform(X[numeric_features]),
                columns=numeric_features,
                index=X.index
            )
    
    # Scale selected features
    if columns_to_standardize:
        if scaler is None:
            scaler = StandardScaler()
            X[columns_to_standardize] = pd.DataFrame(
                scaler.fit_transform(X[columns_to_standardize]),
                columns=columns_to_standardize,
                index=X.index
            )
        else:
            X[columns_to_standardize] = pd.DataFrame(
                scaler.transform(X[columns_to_standardize]),
                columns=columns_to_standardize,
                index=X.index
            )
    
    # Handle categorical features
    if len(categorical_features) > 0:
        if cat_imputer is None:
            cat_imputer = SimpleImputer(strategy='constant', fill_value='unknown')
            X[categorical_features] = pd.DataFrame(
                cat_imputer.fit_transform(X[categorical_features]),
                columns=categorical_features,
                index=X.index
            )
        else:
            X[categorical_features] = pd.DataFrame(
                cat_imputer.transform(X[categorical_features]),
                columns=categorical_features,
                index=X.index
            )
        
        # Target encoding
        target_encode_features = [col for col in target_encode_cols if col in categorical_features]
        if target_encode_features and y is not None:
            if target_encoder is None:
                target_encoder = TargetEncoder(cols=target_encode_features, smoothing=encoding_smoothing)
                X[target_encode_features] = target_encoder.fit_transform(X[target_encode_features], y)
            else:
                X[target_encode_features] = target_encoder.transform(X[target_encode_features])
        
        # One-hot encoding for remaining categorical features
        other_categorical = [col for col in categorical_features if col not in target_encode_features]
        if len(other_categorical) > 0:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_features = encoder.fit_transform(X[other_categorical])
            encoded_feature_names = encoder.get_feature_names_out(other_categorical)
            encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=X.index)
            X = pd.concat([X, encoded_df], axis=1)
            X = X.drop(columns=other_categorical)
    
    logging.info(f"Completed preprocessing. Final shape: {X.shape}")
    return X, num_imputer, cat_imputer, target_encoder, scaler

def find_optimal_clusters(
    X: pd.DataFrame,
    y: pd.Series,
    max_clusters: int = 10,
    features_for_clustering: List[str] = ['depreciation', 'coe', 'dereg_value']
) -> int:
    """
    Find optimal number of clusters using silhouette score.
    """
    logging.info("Starting cluster optimization")
    
    # Verify features exist
    missing_features = [f for f in features_for_clustering if f not in X.columns]
    if missing_features:
        raise ValueError(f"Missing clustering features: {missing_features}")
    
    cluster_features_df = pd.DataFrame(X[features_for_clustering])
    imputer = SimpleImputer(strategy='median')
    cluster_features_clean = imputer.fit_transform(cluster_features_df)
    
    cluster_features = np.column_stack([np.log1p(y), cluster_features_clean])
    
    silhouette_scores = []
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(cluster_features)
        silhouette_avg = silhouette_score(cluster_features, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        logging.info(f"n_clusters = {n_clusters}, silhouette score = {silhouette_avg:.4f}")
    
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    logging.info(f"Selected optimal number of clusters: {optimal_clusters}")
    return optimal_clusters

def create_price_clusters(
    X: pd.DataFrame,
    y: pd.Series,
    n_clusters: int,
    features_for_clustering: List[str] = ['depreciation', 'coe', 'dereg_value']
) -> Tuple[KMeans, np.ndarray, pd.DataFrame]:
    """
    Create price-based clusters using KMeans.
    """
    logging.info(f"Creating {n_clusters} price clusters")
    
    cluster_features_df = pd.DataFrame(X[features_for_clustering])
    imputer = SimpleImputer(strategy='median')
    cluster_features_clean = imputer.fit_transform(cluster_features_df)
    
    price_percentiles = np.percentile(y, np.linspace(0, 100, n_clusters))
    initial_centers = np.column_stack([
        np.log1p(price_percentiles),
        np.percentile(cluster_features_clean, np.linspace(0, 100, n_clusters), axis=0)
    ])
    
    kmeans = KMeans(n_clusters=n_clusters, init=initial_centers, n_init=3, random_state=42)
    cluster_features = np.column_stack([np.log1p(y), cluster_features_clean])
    price_clusters = kmeans.fit_predict(cluster_features)
    
    # Analyze clusters
    cluster_info = []
    for cluster in range(n_clusters):
        cluster_prices = y[price_clusters == cluster]
        cluster_info.append({
            'cluster': cluster,
            'min': cluster_prices.min(),
            'max': cluster_prices.max(),
            'median': cluster_prices.median(),
            'count': len(cluster_prices)
        })
    
    cluster_df = pd.DataFrame(cluster_info)
    logging.info("\nCluster Statistics:")
    logging.info(cluster_df)
    
    kmeans.feature_imputer = imputer
    return kmeans, price_clusters, cluster_df

def train_gradientBoosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Dict[str, Any]
) -> GradientBoostingRegressor:
    """
    Train a GradientBoostingRegressor model.
    """
    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    
    # Calculate validation metrics
    val_predictions = model.predict(X_val)
    val_mse = mean_squared_error(y_val, val_predictions)
    val_r2 = r2_score(y_val, val_predictions)
    
    logging.info(f"Validation RMSE: {np.sqrt(val_mse):.4f}")
    logging.info(f"Validation R2: {val_r2:.4f}")
    
    return model

def predict_cluster(
    X: pd.DataFrame,
    y: Optional[pd.Series],
    kmeans_model: KMeans,
    preprocessors: Dict[str, Any],
    features_for_clustering: List[str] = ['depreciation', 'coe', 'dereg_value']
) -> np.ndarray:
    """
    Predict clusters for new data.
    """
    X_processed, _, _, _, _ = preprocess_features(X, y, **preprocessors)
    
    cluster_features_df = pd.DataFrame(X_processed[features_for_clustering])
    cluster_features_clean = kmeans_model.feature_imputer.transform(cluster_features_df)
    
    dummy_y = np.zeros(len(X)) if y is None else np.log1p(y)
    cluster_features = np.column_stack([dummy_y, cluster_features_clean])
    
    return kmeans_model.predict(cluster_features)

def post_process_predictions(
    predictions: np.ndarray,
    min_price: float = 700,
    max_price: float = 2900000
) -> np.ndarray:
    """
    Post-process predictions by clipping to valid range.
    """
    return np.clip(predictions, min_price, max_price)

def verify_saved_model(model_path: str) -> bool:
    """
    Verify the completeness and integrity of a saved model.
    """
    try:
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        required_keys = ['models', 'kmeans_model', 'cluster_info']
        missing_keys = [key for key in required_keys if key not in loaded_model]
        
        if missing_keys:
            raise ValueError(f"Saved model missing required keys: {missing_keys}")
        
        logging.info("Model verification completed successfully")
        return True
    except Exception as e:
        logging.error(f"Model verification failed: {str(e)}")
        return False

def main():
    """
    Main execution function.
    """
    try:
        np.random.seed(42)
        warnings.filterwarnings('ignore', category=UserWarning)
        
        # Load and preprocess data
        X, y = load_and_preprocess_data('preprocessing/2024-10-21-silan/train_cleaned.csv')
        
        logging.info("\nTarget variable (price) statistics:")
        logging.info(y.describe())
        
        # Cluster optimization and creation
        features_for_clustering = ['depreciation', 'coe', 'dereg_value']
        optimal_clusters = find_optimal_clusters(X, y, max_clusters=3, features_for_clustering=features_for_clustering)
        kmeans_model, price_clusters, cluster_info = create_price_clusters(
            X, y, n_clusters=optimal_clusters,
            features_for_clustering=features_for_clustering
        )
        
        # Model training setup
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        oof_predictions = np.zeros(len(X))
        feature_importance_list = []
        models = []
        
        params = {
            'n_estimators': 3000,
            'learning_rate': 0.05,
            'max_depth': 5,
            'min_samples_split': 20,
            'min_samples_leaf': 15,
            'loss': 'huber',
            'random_state': 42
        }
        
        start_time = time.time()
        
        # Train models for each cluster
        for cluster in range(len(cluster_info)):
            logging.info(f"\nTraining models for Cluster {cluster}")
            X_cluster = X[price_clusters == cluster]
            y_cluster = y[price_clusters == cluster]
            
            cluster_models = []
            
            for fold, (train_index, val_index) in enumerate(kf.split(X_cluster), 1):
                logging.info(f"Processing fold {fold}")
                
                X_train, X_val = X_cluster.iloc[train_index].copy(), X_cluster.iloc[val_index].copy()
                y_train, y_val = y_cluster.iloc[train_index], y_cluster.iloc[val_index]
                
                # Preprocess training and validation data
                X_train_processed, num_imputer, cat_imputer, target_encoder, scaler = preprocess_features(
                    X_train, y_train
                )
                X_val_processed, _, _, _, _ = preprocess_features(
                    X_val, y_val,
                    num_imputer=num_imputer,
                    cat_imputer=cat_imputer,
                    target_encoder=target_encoder,
                    scaler=scaler
                )
                
                # Train model
                model = train_gradientBoosting(X_train_processed, y_train, X_val_processed, y_val, params)
                
                # Generate and store predictions
                fold_predictions = np.expm1(model.predict(X_val_processed))
                oof_predictions[price_clusters == cluster][val_index] = fold_predictions
                
                # Calculate and store feature importance
                importance = model.feature_importances_
                feature_importance = pd.DataFrame({
                    'feature': X_train_processed.columns,
                    'importance': importance
                })
                feature_importance_list.append(feature_importance)
                
                # Store model and preprocessors
                cluster_models.append({
                    'model': model,
                    'preprocessors': {
                        'num_imputer': num_imputer,
                        'cat_imputer': cat_imputer,
                        'target_encoder': target_encoder,
                        'scaler': scaler
                    }
                })
            
            models.append(cluster_models)
        
        # Training completion statistics
        elapsed_time = time.time() - start_time
        logging.info(f"\nTotal training time: {elapsed_time/60:.2f} minutes")
        
        # Process and evaluate out-of-fold predictions
        oof_predictions = post_process_predictions(oof_predictions)
        oof_mse = mean_squared_error(y, oof_predictions)
        oof_r2 = r2_score(y, oof_predictions)
        logging.info(f"Out-of-fold RMSE: {np.sqrt(oof_mse):.4f}")
        logging.info(f"Out-of-fold R2: {oof_r2:.4f}")
        
        # Analyze feature importance
        feature_importance = pd.concat(feature_importance_list).groupby('feature').mean().sort_values('importance', ascending=False)
        logging.info("\nTop 10 important features:")
        logging.info(feature_importance.head(10))
        
        # Save models and related components
        model_save_path = 'gradient_boosting_clustered_models.pkl'
        with open(model_save_path, 'wb') as f:
            pickle.dump({
                'models': models,
                'kmeans_model': kmeans_model,
                'cluster_info': cluster_info,
                'feature_importance': feature_importance
            }, f)
        logging.info(f"Models and preprocessors saved to {model_save_path}")
        
        # Verify saved model
        if not verify_saved_model(model_save_path):
            raise RuntimeError("Model verification failed after saving")
        
        # Generate predictions for test data
        X_test, _ = load_and_preprocess_data('preprocessing/2024-10-21-silan/test_cleaned.csv')
        
        # Predict clusters for test data
        dummy_y_test = np.zeros(len(X_test))
        test_clusters = predict_cluster(
            X_test,
            dummy_y_test,
            kmeans_model,
            models[0][0]['preprocessors'],
            features_for_clustering
        )
        
        # Generate predictions for each cluster
        final_predictions = np.zeros(len(X_test))
        
        for cluster in range(len(cluster_info)):
            cluster_mask = test_clusters == cluster
            X_test_cluster = X_test[cluster_mask]
            
            if len(X_test_cluster) == 0:
                logging.warning(f"No samples in test data for cluster {cluster}")
                continue
            
            cluster_predictions = np.zeros((len(X_test_cluster), len(models[cluster])))
            
            for i, model_dict in enumerate(models[cluster]):
                try:
                    model = model_dict['model']
                    preprocessors = model_dict['preprocessors']
                    
                    X_test_processed, _, _, _, _ = preprocess_features(
                        X_test_cluster,
                        y=None,
                        **preprocessors
                    )
                    cluster_predictions[:, i] = np.expm1(model.predict(X_test_processed))
                except Exception as e:
                    logging.error(f"Error predicting cluster {cluster}, model {i}: {str(e)}")
                    logging.error(f"Shape of X_test_cluster: {X_test_cluster.shape}")
                    logging.error(f"Columns in X_test_cluster: {X_test_cluster.columns}")
                    continue
            
            final_predictions[cluster_mask] = np.mean(cluster_predictions, axis=1)
        
        # Post-process and save predictions
        final_predictions = post_process_predictions(final_predictions)
        
        submission = pd.DataFrame({
            'Id': range(len(final_predictions)),
            'Predicted': np.round(final_predictions).astype(int)
        })
        
        submission_path = '10-27-release_gradientBoosting.csv'
        submission.to_csv(submission_path, index=False)
        logging.info(f"Predictions saved to {submission_path}")
        
        # Log prediction statistics
        logging.info("\nPrediction statistics:")
        logging.info(f"Minimum: {final_predictions.min():.2f}")
        logging.info(f"Maximum: {final_predictions.max():.2f}")
        logging.info(f"Mean: {final_predictions.mean():.2f}")
        logging.info(f"Median: {np.median(final_predictions):.2f}")
        
    except Exception as e:
        logging.error(f"Fatal error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    main()