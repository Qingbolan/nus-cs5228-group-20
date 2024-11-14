import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
import pickle
import time
from category_encoders import TargetEncoder
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WeightedEnsembleL2:
    def __init__(self, n_splits=10, max_clusters=3, random_state=42):
        self.n_splits = n_splits
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.models = []
        self.weights = []
        self.kmeans_model = None
        self.cluster_info = None
        self.preprocessors = []
        
    def load_and_preprocess_data(self, file_path):
        data = pd.read_csv(file_path)
        if 'Unnamed: 0' in data.columns:
            data = data.drop(columns='Unnamed: 0')
        
        X = data.drop('price', axis=1) if 'price' in data.columns else data
        y = data['price'] if 'price' in data.columns else None
        
        logging.info(f"Columns in {file_path}: {X.columns}")
        return X, y

    def preprocess_features(self, X, y=None, num_imputer=None, cat_imputer=None, 
                          target_encoder=None, scaler=None, 
                          target_encode_cols=['make', 'model']):
        X = X.copy()
        
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        # Numeric imputation
        if num_imputer is None:
            num_imputer = SimpleImputer(strategy='median')
            X[numeric_features] = pd.DataFrame(num_imputer.fit_transform(X[numeric_features]), 
                                             columns=numeric_features, index=X.index)
        else:
            X[numeric_features] = pd.DataFrame(num_imputer.transform(X[numeric_features]), 
                                             columns=numeric_features, index=X.index)
        
        # Categorical feature processing
        if len(categorical_features) > 0:
            if cat_imputer is None:
                cat_imputer = SimpleImputer(strategy='constant', fill_value='unknown')
                X[categorical_features] = pd.DataFrame(cat_imputer.fit_transform(X[categorical_features]), 
                                                     columns=categorical_features, index=X.index)
            else:
                X[categorical_features] = pd.DataFrame(cat_imputer.transform(X[categorical_features]), 
                                                     columns=categorical_features, index=X.index)
            
            # Target encoding
            target_encode_features = [col for col in target_encode_cols if col in categorical_features]
            if target_encode_features and y is not None:
                if target_encoder is None:
                    target_encoder = TargetEncoder(cols=target_encode_features)
                    X[target_encode_features] = target_encoder.fit_transform(X[target_encode_features], y)
                else:
                    X[target_encode_features] = target_encoder.transform(X[target_encode_features])
            
            # One-hot encoding for remaining categorical features
            other_categorical = [col for col in categorical_features if col not in target_encode_features]
            if other_categorical:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_features = encoder.fit_transform(X[other_categorical])
                encoded_feature_names = encoder.get_feature_names_out(other_categorical)
                encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=X.index)
                X = pd.concat([X.drop(columns=other_categorical), encoded_df], axis=1)
        
        return X, num_imputer, cat_imputer, target_encoder, scaler
    
    def create_clusters(self, X, y, features_for_clustering=['depreciation', 'coe', 'dereg_value']):
        cluster_features_df = pd.DataFrame(X[features_for_clustering])
        imputer = SimpleImputer(strategy='median')
        cluster_features_clean = imputer.fit_transform(cluster_features_df)
        
        # Combine with log-transformed price
        cluster_features = np.column_stack([np.log1p(y), cluster_features_clean])
        
        # Find optimal number of clusters
        silhouette_scores = []
        for n_clusters in range(2, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=self.random_state)
            cluster_labels = kmeans.fit_predict(cluster_features)
            silhouette_avg = silhouette_score(cluster_features, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
        
        # Create final clustering model
        initial_centers = np.column_stack([
            np.log1p(np.percentile(y, np.linspace(0, 100, optimal_clusters))),
            np.percentile(cluster_features_clean, np.linspace(0, 100, optimal_clusters), axis=0)
        ])
        
        self.kmeans_model = KMeans(n_clusters=optimal_clusters, init=initial_centers, n_init=3, random_state=self.random_state)
        self.kmeans_model.feature_imputer = imputer
        price_clusters = self.kmeans_model.fit_predict(cluster_features)
        
        # Store cluster information
        cluster_info = []
        for cluster in range(optimal_clusters):
            cluster_prices = y[price_clusters == cluster]
            cluster_info.append({
                'cluster': cluster,
                'min': cluster_prices.min(),
                'max': cluster_prices.max(),
                'median': cluster_prices.median(),
                'count': len(cluster_prices)
            })
        self.cluster_info = pd.DataFrame(cluster_info)
        
        return price_clusters

    def train_model(self, X_train, y_train, X_val, y_val):
        params = {
            'objective': 'regression',
            'eta': 0.03, 'max_depth': 5, 'min_child_weight': 5, 'subsample': 0.7, 'colsample_bytree': 0.9, 'gamma': 0, 'eval_metric': 'rmse', 'booster': 'gbtree', 'seed': 42}
        
        train_data = lgb.Dataset(X_train, label=np.log1p(y_train))
        val_data = lgb.Dataset(X_val, label=np.log1p(y_val), reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid']
        )
        
        return model
    
    def fit(self, X, y):
        # Create clusters
        features_for_clustering = ['depreciation', 'coe', 'dereg_value']
        price_clusters = self.create_clusters(X, y, features_for_clustering)
        
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        # Train models for each cluster
        for cluster in range(len(self.cluster_info)):
            X_cluster = X[price_clusters == cluster]
            y_cluster = y[price_clusters == cluster]
            
            cluster_models = []
            cluster_weights = []
            cluster_preprocessors = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_cluster)):
                X_train, X_val = X_cluster.iloc[train_idx], X_cluster.iloc[val_idx]
                y_train, y_val = y_cluster.iloc[train_idx], y_cluster.iloc[val_idx]
                
                # Preprocess data
                X_train_processed, num_imputer, cat_imputer, target_encoder, scaler = self.preprocess_features(X_train, y_train)
                X_val_processed, _, _, _, _ = self.preprocess_features(X_val, y_val, num_imputer, cat_imputer, target_encoder, scaler)
                
                # Train model
                model = self.train_model(X_train_processed, y_train, X_val_processed, y_val)
                
                # Calculate weight based on validation performance
                val_pred = np.expm1(model.predict(X_val_processed))
                weight = 1 / mean_squared_error(y_val, val_pred)
                
                cluster_models.append(model)
                cluster_weights.append(weight)
                cluster_preprocessors.append({
                    'num_imputer': num_imputer,
                    'cat_imputer': cat_imputer,
                    'target_encoder': target_encoder,
                    'scaler': scaler
                })
            
            # Normalize weights
            cluster_weights = np.array(cluster_weights) / np.sum(cluster_weights)
            
            self.models.append(cluster_models)
            self.weights.append(cluster_weights)
            self.preprocessors.append(cluster_preprocessors)
        
        return self

    def predict(self, X):
        dummy_y = np.zeros(len(X))
        test_clusters = self.predict_clusters(X, dummy_y)
        
        predictions = np.zeros(len(X))
        
        for cluster in range(len(self.cluster_info)):
            cluster_mask = test_clusters == cluster
            X_cluster = X[cluster_mask]
            
            if len(X_cluster) == 0:
                continue
            
            cluster_predictions = np.zeros((len(X_cluster), len(self.models[cluster])))
            
            for i, (model, preprocessors) in enumerate(zip(self.models[cluster], self.preprocessors[cluster])):
                X_processed, _, _, _, _ = self.preprocess_features(X_cluster, **preprocessors)
                cluster_predictions[:, i] = np.expm1(model.predict(X_processed)) * self.weights[cluster][i]
            
            predictions[cluster_mask] = np.sum(cluster_predictions, axis=1)
        
        return np.clip(predictions, 700, 2900000)

    def predict_clusters(self, X, y=None, features_for_clustering=['depreciation', 'coe', 'dereg_value']):
        cluster_features_df = pd.DataFrame(X[features_for_clustering])
        cluster_features_clean = self.kmeans_model.feature_imputer.transform(cluster_features_df)
        
        cluster_features = np.column_stack([
            np.log1p(y) if y is not None else np.zeros(len(X)),
            cluster_features_clean
        ])
        
        return self.kmeans_model.predict(cluster_features)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'weights': self.weights,
                'kmeans_model': self.kmeans_model,
                'cluster_info': self.cluster_info,
                'preprocessors': self.preprocessors
            }, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls()
        instance.models = data['models']
        instance.weights = data['weights']
        instance.kmeans_model = data['kmeans_model']
        instance.cluster_info = data['cluster_info']
        instance.preprocessors = data['preprocessors']
        
        return instance

def main():
    # Initialize the ensemble
    ensemble = WeightedEnsembleL2(n_splits=10, max_clusters=3, random_state=42)
    
    # Load and train
    X_train, y_train = ensemble.load_and_preprocess_data('l2_train.csv')
    ensemble.fit(X_train, y_train)
    
    # Save the trained model
    ensemble.save('weighted_ensemble_l2_model.pkl')
    
    # Make predictions on test data
    X_test, _ = ensemble.load_and_preprocess_data('l2_test.csv')
    predictions = ensemble.predict(X_test)
    
    # Create submission
    submission = pd.DataFrame({
        'Id': range(len(predictions)),
        'Predicted': np.round(predictions).astype(int)
    })
    submission.to_csv('weighted_ensemble_l2_predictions.csv', index=False)
    
    logging.info("\nPrediction statistics:")
    logging.info(f"Minimum: {predictions.min()}")
    logging.info(f"Maximum: {predictions.max()}")
    logging.info(f"Mean: {predictions.mean()}")
    logging.info(f"Median: {np.median(predictions)}")

if __name__ == '__main__':
    main()