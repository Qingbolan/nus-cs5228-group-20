import os
import logging
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.svm import SVR
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)

def load_and_preprocess_data(file_path: str) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Load and perform initial preprocessing of data."""
    if not os.path.exists(file_path):
        logging.error(f"Data file not found: {file_path}")
        raise FileNotFoundError(f"Data file not found: {file_path}")

    try:
        data = pd.read_csv(file_path)
        logging.info(f"Successfully loaded data from {file_path}. Shape: {data.shape}")
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {str(e)}")
        raise

    # Drop all 'Unnamed' columns
    unnamed_cols = [col for col in data.columns if "Unnamed" in col]
    if unnamed_cols:
        data = data.drop(columns=unnamed_cols)
        logging.info(f"Dropped columns: {unnamed_cols}")

    X = data.drop('price', axis=1) if 'price' in data.columns else data
    y = data['price'] if 'price' in data.columns else None

    return X, y

def create_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """Create a preprocessing pipeline using ColumnTransformer."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    return preprocessor

def find_optimal_clusters(
    X: pd.DataFrame,
    y: pd.Series,
    max_clusters: int = 10,
    features_for_clustering: List[str] = ['depreciation', 'coe', 'dereg_value']
) -> int:
    """Find optimal number of clusters using silhouette score."""
    logging.info("Starting cluster optimization")

    missing_features = [f for f in features_for_clustering if f not in X.columns]
    if missing_features:
        logging.error(f"Missing clustering features: {missing_features}")
        raise ValueError(f"Missing clustering features: {missing_features}")

    cluster_features_df = X[features_for_clustering]
    imputer = SimpleImputer(strategy='median')
    cluster_features_clean = imputer.fit_transform(cluster_features_df)

    scaler = RobustScaler()
    cluster_features_scaled = scaler.fit_transform(cluster_features_clean)
    cluster_features = np.column_stack([np.log1p(y), cluster_features_scaled])

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
    """Create price-based clusters."""
    logging.info(f"Creating {n_clusters} price clusters")

    cluster_features_df = X[features_for_clustering]
    imputer = SimpleImputer(strategy='median')
    cluster_features_clean = imputer.fit_transform(cluster_features_df)

    scaler = RobustScaler()
    cluster_features_scaled = scaler.fit_transform(cluster_features_clean)

    # Combine with the target
    if y is not None:
        with np.errstate(divide='ignore'):
            y_log = np.log1p(y)
            y_log[np.isneginf(y_log)] = 0  # Handle log(0) if any
    else:
        y_log = np.zeros(len(X))
        logging.warning("Target y is None. Using dummy price values for clustering.")

    cluster_features = np.column_stack([y_log, cluster_features_scaled])

    # Initialize KMeans with k-means++ and fixed random_state for reproducibility
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=20, random_state=42)
    price_clusters = kmeans.fit_predict(cluster_features)

    # Collect cluster statistics
    cluster_info = []
    for cluster in range(n_clusters):
        cluster_prices = y[price_clusters == cluster] if y is not None else pd.Series([])
        if not cluster_prices.empty:
            cluster_info.append({
                'cluster': cluster,
                'min': cluster_prices.min(),
                'max': cluster_prices.max(),
                'median': cluster_prices.median(),
                'count': len(cluster_prices)
            })
        else:
            cluster_info.append({
                'cluster': cluster,
                'min': None,
                'max': None,
                'median': None,
                'count': 0
            })

    cluster_df = pd.DataFrame(cluster_info)
    logging.info("Cluster Statistics:\n%s", cluster_df)

    # Attach preprocessing objects for later use
    kmeans.feature_imputer = imputer
    kmeans.feature_scaler = scaler

    return kmeans, price_clusters, cluster_df

def predict_cluster(
    X: pd.DataFrame,
    y: Optional[pd.Series],
    kmeans_model: KMeans,
    features_for_clustering: List[str] = ['depreciation', 'coe', 'dereg_value']
) -> np.ndarray:
    """Predict clusters for new data."""
    cluster_features_df = X[features_for_clustering]
    cluster_features_clean = kmeans_model.feature_imputer.transform(cluster_features_df)
    cluster_features_scaled = kmeans_model.feature_scaler.transform(cluster_features_clean)

    if y is not None:
        with np.errstate(divide='ignore'):
            y_log = np.log1p(y)
            y_log[np.isneginf(y_log)] = 0  # Handle log(0) if any
        cluster_features = np.column_stack([y_log, cluster_features_scaled])
    else:
        # For test data, use dummy price value
        dummy_price = np.zeros(len(X))
        cluster_features = np.column_stack([dummy_price, cluster_features_scaled])

    return kmeans_model.predict(cluster_features)

def optimize_svr(model_pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> SVR:
    """Perform hyperparameter tuning for SVR using GridSearchCV."""
    param_grid = {
        'svr__C': [50, 100, 150],
        'svr__epsilon': [0.05, 0.1, 0.2],
        'svr__gamma': ['scale', 'auto']
    }
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X, y)
    logging.info(f"Best SVR Params: {grid_search.best_params_}")
    return grid_search.best_estimator_

def main():
    """Main execution function with optimized SVR training."""
    try:
        np.random.seed(42)

        # File paths
        train_path = 'preprocessing/2024-10-21-silan/train_cleaned.csv'
        test_path = 'preprocessing/2024-10-21-silan/test_cleaned.csv'
        model_save_path = 'optimized_svr_models.joblib'
        prediction_save_path = 'optimized_svr_predictions.csv'

        # Load training data
        X, y = load_and_preprocess_data(train_path)

        # Identify feature types
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        logging.info(f"Numeric features: {numeric_features}")
        logging.info(f"Categorical features: {categorical_features}")

        # Clustering
        features_for_clustering = ['depreciation', 'coe', 'dereg_value']
        optimal_clusters = find_optimal_clusters(
            X, y, max_clusters=10, features_for_clustering=features_for_clustering
        )
        kmeans_model, price_clusters, cluster_info = create_price_clusters(
            X, y, n_clusters=optimal_clusters, features_for_clustering=features_for_clustering
        )

        # Define dynamic price ranges based on clusters
        price_ranges = []
        sorted_clusters = cluster_info.sort_values(by='median').reset_index(drop=True)
        for _, row in sorted_clusters.iterrows():
            if row['median'] is not None:
                if row['median'] < sorted_clusters['median'].median():
                    price_ranges.append('low')
                elif row['median'] < sorted_clusters['median'].quantile(0.75):
                    price_ranges.append('medium')
                else:
                    price_ranges.append('high')
            else:
                price_ranges.append('medium')  # Default range

        logging.info(f"Price ranges assigned to clusters: {price_ranges}")

        # Preprocessor
        preprocessor = create_preprocessor(numeric_features, categorical_features)

        # Train SVR models for each cluster
        cluster_models = []
        for cluster in range(optimal_clusters):
            logging.info(f"Training model for Cluster {cluster} with price range '{price_ranges[cluster]}'")
            mask = price_clusters == cluster
            X_cluster = X[mask]
            y_cluster = y[mask]

            if X_cluster.empty:
                logging.warning(f"No data in Cluster {cluster}. Skipping model training.")
                cluster_models.append(None)
                continue

            # Define pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('svr', SVR())
            ])

            # Perform hyperparameter optimization
            optimized_model = optimize_svr(pipeline, X_cluster, np.log1p(y_cluster))
            cluster_models.append(optimized_model)

        # Save the trained models and clustering information
        joblib.dump({
            'models': cluster_models,
            'kmeans': kmeans_model,
            'cluster_info': cluster_info,
            'preprocessor': preprocessor
        }, model_save_path)
        logging.info(f"Saved trained models to {model_save_path}")

        # Load test data
        X_test, _ = load_and_preprocess_data(test_path)

        # Predict clusters for test data
        test_clusters = predict_cluster(X_test, None, kmeans_model, features_for_clustering)
        logging.info("Predicted clusters for test data.")

        # Initialize array for final predictions
        final_predictions = np.zeros(len(X_test))

        for cluster in range(optimal_clusters):
            cluster_mask = test_clusters == cluster
            if not cluster_mask.any():
                logging.warning(f"No test samples in Cluster {cluster}. Skipping prediction for this cluster.")
                continue

            X_test_cluster = X_test[cluster_mask]
            model = cluster_models[cluster]

            if model is None:
                logging.warning(f"No model found for Cluster {cluster}. Skipping prediction.")
                continue

            # Predict and inverse transform
            predictions_log = model.predict(X_test_cluster)
            predictions = np.expm1(predictions_log)
            final_predictions[cluster_mask] = predictions

        # Post-process predictions
        final_predictions = np.clip(final_predictions, 700, 2_900_000).round().astype(int)

        # Save predictions
        submission = pd.DataFrame({
            'Id': np.arange(len(final_predictions)),
            'Predicted': final_predictions
        })
        submission.to_csv(prediction_save_path, index=False)
        logging.info(f"Saved predictions to {prediction_save_path}")

    except Exception as e:
        logging.exception(f"An error occurred during execution: {str(e)}")
        raise

if __name__ == '__main__':
    main()