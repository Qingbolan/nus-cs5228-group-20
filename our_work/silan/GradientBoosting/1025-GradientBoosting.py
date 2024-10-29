import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold 
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle
import time
import logging
from datetime import datetime

# Configure logging to display INFO level messages and above
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OptimizedCarPricePredictor:
    def __init__(self, n_folds=5, seed=42):
        self.n_folds = n_folds
        self.seed = seed
        self.initial_numeric_features = [
            'manufactured', 'curb_weight', 'power', 'engine_cap', 
            'no_of_owners', 'depreciation', 'coe', 'road_tax', 
            'dereg_value', 'omv', 'arf', 'vehicle_age'
        ]
        self.initial_bool_features = [
            'vehicle_type_bus_mini_bus', 'vehicle_type_hatchback',
            'vehicle_type_luxury_sedan', 'vehicle_type_midsized_sedan',
            'vehicle_type_mpv', 'vehicle_type_others', 'vehicle_type_sports_car',
            'vehicle_type_stationwagon', 'vehicle_type_suv', 'vehicle_type_truck',
            'vehicle_type_van', 'transmission_type_auto', 'transmission_type_manual',
            'almost_new_car', 'coe_car', 'consignment_car', 'direct_owner_sale',
            'electric_cars', 'hybrid_cars', 'imported_used_vehicle', 'low_mileage_car',
            'opc_car', 'parf_car', 'premium_ad_car', 'rare_and_exotic',
            'sgcarmart_warranty_cars', 'sta_evaluated_car', 'vintage_cars'
        ]
        self.feature_mapping = None
        self.numeric_features = None
        self.bool_features = None
        self.feature_columns = None
        self.model_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.01,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'max_depth': 8,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbose': -1
        }
    
    def clean_feature_names(self, X):
        """
        Clean feature names by removing or replacing special characters.
        
        Args:
            X (pd.DataFrame): The input DataFrame with original feature names.
        
        Returns:
            pd.DataFrame: DataFrame with cleaned feature names.
            dict: Mapping from original feature names to cleaned feature names.
        """
        X = X.copy()
        feature_mapping = {}
        for col in X.columns:
            new_name = (col.lower()
                        .replace(' ', '_')
                        .replace('-', '_')
                        .replace('(', '')
                        .replace(')', '')
                        .replace('[', '')
                        .replace(']', '')
                        .replace('<', '_lt_')
                        .replace('>', '_gt_')
                        .replace('=', '_eq_')
                        .replace('/', '_')
                        .replace('\\', '_')
                        .replace('+', '_plus_')
                        .replace('&', '_and_')
                        .replace('%', '_pct_')
                        .replace('$', '_dollar_')
                        .replace('@', '_at_')
                        .replace('!', '_')
                        .replace('?', '_')
                        .replace('*', '_star_')
                        .replace(':', '_')
                        .replace(';', '_')
                        .replace(',', '_')
                        .replace('.', '_'))
            
            if not new_name[0].isalpha():
                new_name = 'f_' + new_name
            feature_mapping[col] = new_name
        
        X.columns = [feature_mapping[col] for col in X.columns]
        logging.info("Feature names cleaned and mapped.")
        return X, feature_mapping

    def initialize_feature_names(self, df):
        """
        Initialize feature names by cleaning and mapping them.
        
        Args:
            df (pd.DataFrame): The input DataFrame with original feature names.
        
        Returns:
            pd.DataFrame: DataFrame with cleaned feature names.
        """
        cleaned_df, self.feature_mapping = self.clean_feature_names(df)
        
        # Update numeric and boolean features with cleaned names
        self.numeric_features = [self.feature_mapping.get(f, f) for f in self.initial_numeric_features]
        self.bool_features = [self.feature_mapping.get(f, f) for f in self.initial_bool_features]
        
        # Verify presence of all required boolean features
        missing_bool_features = [f for f in self.bool_features if f not in cleaned_df.columns]
        if missing_bool_features:
            for feature in missing_bool_features:
                logging.warning(f"Boolean feature '{feature}' is missing. Adding it with default value 0.")
                cleaned_df[feature] = 0  # Assign default value
        else:
            logging.info("All initial boolean features are present.")
        
        return cleaned_df

    def create_features(self, df):
        """
        Create additional features based on existing data.
        
        Args:
            df (pd.DataFrame): The input DataFrame with cleaned feature names.
        
        Returns:
            pd.DataFrame: DataFrame with new features added.
        """
        data = df.copy()
        
        # List of required boolean features for feature engineering
        required_bool_features = [
            'vehicle_type_bus_mini_bus', 'vehicle_type_hatchback',
            'vehicle_type_luxury_sedan', 'vehicle_type_midsized_sedan',
            'vehicle_type_mpv', 'vehicle_type_others', 'vehicle_type_sports_car',
            'vehicle_type_stationwagon', 'vehicle_type_suv', 'vehicle_type_truck',
            'vehicle_type_van', 'transmission_type_auto', 'transmission_type_manual',
            'almost_new_car', 'coe_car', 'consignment_car', 'direct_owner_sale',
            'electric_cars', 'hybrid_cars', 'imported_used_vehicle', 'low_mileage_car',
            'opc_car', 'parf_car', 'premium_ad_car', 'rare_and_exotic',
            'sgcarmart_warranty_cars', 'sta_evaluated_car', 'vintage_cars'
        ]
        
        # Ensure all required boolean features exist
        for feature in required_bool_features:
            if feature not in data.columns:
                logging.warning(f"Missing boolean feature '{feature}'. Adding it with default value 0.")
                data[feature] = 0  # Assign default value
        
        # Handle possible division by zero using np.maximum
        data['power_weight_ratio'] = data['power'] / np.maximum(data['curb_weight'], 1)
        data['engine_efficiency'] = data['power'] / np.maximum(data['engine_cap'], 1)
        data['price_per_year'] = data['depreciation'] / np.maximum(data['vehicle_age'], 1)
        
        # Cost-related features
        data['total_cost'] = data['omv'] + data['arf'] + data['coe']
        data['value_retention'] = data['dereg_value'] / np.maximum(data['total_cost'], 1)
        data['monthly_depreciation'] = data['depreciation'] / 12
        data['maintenance_cost'] = data['road_tax'] * data['vehicle_age']
        
        # Vehicle age related features
        data['age_squared'] = data['vehicle_age'] ** 2
        data['age_cubed'] = data['vehicle_age'] ** 3
        data['manufactured_year'] = datetime.now().year - data['vehicle_age']
        
        # Composite cost features
        data['cost_per_power'] = data['total_cost'] / np.maximum(data['power'], 1)
        data['depreciation_rate'] = data['depreciation'] / np.maximum(data['total_cost'], 1)
        
        # Luxury score calculation with handling for missing features
        luxury_components = []
        if 'vehicle_type_luxury_sedan' in data.columns:
            luxury_components.append((data['vehicle_type_luxury_sedan'] == 1) * 3)
        else:
            luxury_components.append(0)
            logging.warning("Missing 'vehicle_type_luxury_sedan'. Assigning 0 for luxury score component.")
        
        if 'rare_and_exotic' in data.columns:
            luxury_components.append((data['rare_and_exotic'] == 1) * 2)
        else:
            luxury_components.append(0)
            logging.warning("Missing 'rare_and_exotic'. Assigning 0 for luxury score component.")
        
        if 'premium_ad_car' in data.columns:
            luxury_components.append((data['premium_ad_car'] == 1))
        else:
            luxury_components.append(0)
            logging.warning("Missing 'premium_ad_car'. Assigning 0 for luxury score component.")
        
        data['luxury_score'] = sum(luxury_components)
        
        logging.info("Additional features created successfully.")
        return data

    def handle_outliers(self, df, columns):
        """
        Handle outliers in specified columns by clipping.
        
        Args:
            df (pd.DataFrame): The input DataFrame.
            columns (list): List of column names to handle outliers.
        
        Returns:
            pd.DataFrame: DataFrame with outliers handled.
        """
        data = df.copy()
        for col in columns:
            if col in data.columns:
                q1 = data[col].quantile(0.01)
                q3 = data[col].quantile(0.99)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
                logging.info(f"Outliers in '{col}' handled by clipping.")
            else:
                logging.warning(f"Column '{col}' not found for outlier handling.")
        return data

    def find_optimal_clusters(self, X, y):
        """
        Find the optimal number of clusters using silhouette scores.
        
        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable.
        
        Returns:
            np.ndarray: Cluster labels for each sample.
        """
        self.cluster_features = ['depreciation', 'coe', 'dereg_value', 'total_cost', 'value_retention']
        available_cluster_features = [feature for feature in self.cluster_features if feature in X.columns]
        
        if not available_cluster_features:
            logging.error("No cluster features available for clustering.")
            raise ValueError("No cluster features available for clustering.")
        
        cluster_features = pd.DataFrame(X[available_cluster_features])
        
        self.imputer = SimpleImputer(strategy='median')
        cluster_features_clean = self.imputer.fit_transform(cluster_features)
        
        price_scaler = StandardScaler()
        if 'price' in y.name:
            y_scaled = price_scaler.fit_transform(np.log1p(y.values.reshape(-1, 1))).flatten()
        else:
            y_scaled = price_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
        
        cluster_features_with_price = np.column_stack([cluster_features_clean, y_scaled])
        
        silhouette_scores = []
        max_clusters = 3
        
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=10)
            cluster_labels = kmeans.fit_predict(cluster_features_with_price)
            score = silhouette_score(cluster_features_with_price, cluster_labels)
            silhouette_scores.append(score)
            logging.info(f"Clusters: {n_clusters}, Silhouette Score: {score:.4f}")
        
        if not silhouette_scores:
            logging.error("Silhouette scores could not be computed.")
            raise ValueError("Silhouette scores could not be computed.")
        
        optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
        
        # Fit KMeans without price feature for final clustering
        self.kmeans_no_price = KMeans(n_clusters=optimal_clusters, random_state=self.seed, n_init=10)
        self.kmeans_no_price.fit(cluster_features_clean)
        
        logging.info(f"Optimal number of clusters determined: {optimal_clusters}")
        return self.kmeans_no_price.labels_

    def train_model(self, X, y, clusters):
        """
        Train LightGBM models for each cluster with cross-validation.
        
        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable.
            clusters (np.ndarray): Cluster labels.
        
        Returns:
            list: Trained models for each cluster.
            np.ndarray: Out-of-fold predictions.
            pd.Series: Feature importance scores.
        """
        models = []
        oof_predictions = np.zeros(len(y))
        feature_importance_list = []
        
        unique_clusters = np.unique(clusters)
        logging.info(f"Training models for {len(unique_clusters)} clusters.")
        
        for cluster in unique_clusters:
            logging.info(f"\nTraining models for Cluster {cluster}")
            cluster_mask = clusters == cluster
            X_cluster = X[cluster_mask]
            y_cluster = y[cluster_mask]
            
            cluster_params = self.model_params.copy()
            if len(X_cluster) < 1000:
                cluster_params.update({
                    'learning_rate': 0.01,
                    'num_leaves': 20,
                    'min_child_samples': 10
                })
                logging.info(f"Adjusted model parameters for smaller cluster with {len(X_cluster)} samples.")
            
            cluster_models = []
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_cluster), 1):
                X_train, X_val = X_cluster.iloc[train_idx], X_cluster.iloc[val_idx]
                y_train, y_val = y_cluster.iloc[train_idx], y_cluster.iloc[val_idx]
                
                # Log fold information
                logging.info(f"Cluster {cluster} - Fold {fold}: Training on {len(X_train)} samples, validating on {len(X_val)} samples.")
                
                # Prepare LightGBM datasets
                train_data = lgb.Dataset(X_train, np.log1p(y_train))
                val_data = lgb.Dataset(X_val, np.log1p(y_val), reference=train_data)
                
                # Train the model
                model = lgb.train(
                    cluster_params,
                    train_data,
                    valid_sets=[train_data, val_data],
                    num_boost_round=2000,
                    # early_stopping_rounds=50,
                    # verbose_eval=False
                )
                
                # Predict and store out-of-fold predictions
                fold_preds = np.expm1(model.predict(X_val, num_iteration=model.best_iteration))
                oof_predictions[cluster_mask][val_idx] = fold_preds
                
                # Collect feature importance
                importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importance()
                })
                feature_importance_list.append(importance)
                
                # Append model
                cluster_models.append(model)
                
                # Calculate and log RMSE for the fold
                fold_rmse = np.sqrt(mean_squared_error(y_val, fold_preds))
                logging.info(f"Cluster {cluster} - Fold {fold} - RMSE: {fold_rmse:,.2f}")
            
            models.append(cluster_models)
            logging.info(f"Completed training for Cluster {cluster}")
        
        # Aggregate feature importance
        feature_importance = pd.concat(feature_importance_list)
        feature_importance = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
        logging.info("\nTop 10 Important Features:")
        logging.info(feature_importance.head(10))
        
        return models, oof_predictions, feature_importance

    def preprocess_features(self, df, is_train=True):
        """
        Preprocess features including cleaning, feature engineering, and scaling.
        
        Args:
            df (pd.DataFrame): The input DataFrame.
            is_train (bool): Indicates whether the DataFrame is for training.
        
        Returns:
            pd.DataFrame: Preprocessed feature matrix.
        """
        data = df.copy()
        
        # Clean and map feature names
        if is_train:
            data = self.initialize_feature_names(data)
            logging.info("Initialized and cleaned feature names for training data.")
        else:
            if self.feature_mapping is None:
                logging.error("Feature mapping not found. Must fit model first.")
                raise ValueError("Feature mapping not found. Must fit model first.")
            data.columns = [self.feature_mapping.get(col, col) for col in data.columns]
            logging.info("Cleaned and mapped feature names for test data.")
        
        # Create new features
        data = self.create_features(data)
        
        # Handle outliers for numeric features
        numeric_columns = self.numeric_features
        data = self.handle_outliers(data, numeric_columns)
        
        if is_train:
            # Store feature columns for future reference
            self.feature_columns = data.columns.tolist()
            logging.info("Stored feature columns for training data.")
        else:
            # Ensure test data has the same columns as training data
            missing_cols = set(self.feature_columns) - set(data.columns)
            extra_cols = set(data.columns) - set(self.feature_columns)
            
            # Add missing columns with default value 0
            for col in missing_cols:
                data[col] = 0
                logging.warning(f"Adding missing column to test data: '{col}' with default value 0.")
            
            # Remove extra columns
            for col in extra_cols:
                data = data.drop(col, axis=1)
                logging.warning(f"Removing extra column from test data: '{col}'.")
            
            # Ensure the order of columns matches training data
            data = data[self.feature_columns]
            logging.info("Aligned test data columns with training data.")
        
        # Standardize numerical features
        scale_features = [f for f in ['curb_weight', 'power', 'engine_cap', 'depreciation'] 
                         if f in data.columns]
        
        if is_train:
            self.scaler = StandardScaler()
            data[scale_features] = self.scaler.fit_transform(data[scale_features])
            logging.info("Fitted scaler and transformed training numerical features.")
        else:
            data[scale_features] = self.scaler.transform(data[scale_features])
            logging.info("Transformed test numerical features using fitted scaler.")
        
        return data

    def predict(self, models, X):
        """
        Generate predictions using the trained models.
        
        Args:
            models (list): Trained models for each cluster.
            X (pd.DataFrame): Preprocessed feature matrix for prediction.
        
        Returns:
            np.ndarray: Predicted target values.
        """
        # Extract clustering features
        available_cluster_features = [feature for feature in self.cluster_features if feature in X.columns]
        cluster_features = pd.DataFrame(X[available_cluster_features])
        
        # Impute missing clustering features
        cluster_features_clean = self.imputer.transform(cluster_features)
        
        # Predict clusters
        clusters = self.kmeans_no_price.predict(cluster_features_clean)
        logging.info("Assigned clusters for prediction data.")
        
        predictions = np.zeros(len(X))
        
        for cluster_idx, cluster_models in enumerate(models):
            cluster_mask = clusters == cluster_idx
            if not np.any(cluster_mask):
                logging.warning(f"No samples found for Cluster {cluster_idx} during prediction.")
                continue
            
            X_cluster = X[cluster_mask]
            cluster_preds = np.zeros((len(X_cluster), len(cluster_models)))
            
            for model_idx, model in enumerate(cluster_models):
                cluster_preds[:, model_idx] = np.expm1(model.predict(X_cluster, num_iteration=model.best_iteration))
                logging.debug(f"Model {model_idx} predictions for Cluster {cluster_idx} completed.")
            
            # Average predictions across models
            predictions[cluster_mask] = cluster_preds.mean(axis=1)
            logging.info(f"Predictions for Cluster {cluster_idx} completed.")
        
        return predictions

    def post_process_predictions(self, predictions, y_train):
        """
        Post-process predictions by clipping to a sensible range.
        
        Args:
            predictions (np.ndarray): Raw predictions.
            y_train (pd.Series): Training target variable.
        
        Returns:
            np.ndarray: Post-processed predictions.
        """
        min_price = y_train.min()
        max_price = y_train.max()
        predictions = np.clip(predictions, min_price, max_price)
        
        q1 = np.percentile(predictions, 1)
        q3 = np.percentile(predictions, 99)
        iqr = q3 - q1
        lower_bound = max(q1 - 1.5 * iqr, min_price)
        upper_bound = min(q3 + 1.5 * iqr, max_price)
        
        predictions = np.clip(predictions, lower_bound, upper_bound)
        logging.info("Post-processed predictions by clipping to remove extreme values.")
        return predictions

    def fit_predict(self, train_df, test_df):
        """
        Main workflow to train models and generate predictions.
        
        Args:
            train_df (pd.DataFrame): Training DataFrame.
            test_df (pd.DataFrame): Testing DataFrame.
        
        Returns:
            np.ndarray: Out-of-fold predictions for training data.
            np.ndarray: Predictions for test data.
            pd.Series: Feature importance scores.
        """
        start_time = time.time()
        logging.info(f"Initial train features: {len(train_df.columns)} columns.")
        logging.info(f"Initial test features: {len(test_df.columns)} columns.")
        
        # Preprocess training data
        X_train = self.preprocess_features(train_df, is_train=True)
        y_train = train_df['price']
        logging.info(f"Processed training features: {len(X_train.columns)} columns.")
        
        # Perform clustering
        clusters = self.find_optimal_clusters(X_train, y_train)
        logging.info("Clustering of training data completed.")
        
        # Train models
        models, oof_predictions, feature_importance = self.train_model(X_train, y_train, clusters)
        logging.info("Model training completed.")
        
        # Post-process out-of-fold predictions
        oof_predictions = self.post_process_predictions(oof_predictions, y_train)
        oof_rmse = np.sqrt(mean_squared_error(y_train, oof_predictions))
        oof_r2 = r2_score(y_train, oof_predictions)
        
        logging.info(f"\nOut-of-fold RMSE: {oof_rmse:,.2f}")
        logging.info(f"Out-of-fold R2: {oof_r2:.4f}")
        
        # Preprocess test data and generate predictions
        X_test = self.preprocess_features(test_df, is_train=False)
        test_predictions = self.predict(models, X_test)
        test_predictions = self.post_process_predictions(test_predictions, y_train)
        
        elapsed_time = time.time() - start_time
        logging.info(f"\nTotal training and prediction time: {elapsed_time/60:.2f} minutes.")
        
        return oof_predictions, test_predictions, feature_importance

    def save_model_and_results(self, predictor, feature_importance, oof_predictions, test_predictions, train_df):
        """
        Save the trained model, feature importance, predictions, and performance metrics.
        
        Args:
            predictor (OptimizedCarPricePredictor): The trained predictor instance.
            feature_importance (pd.Series): Feature importance scores.
            oof_predictions (np.ndarray): Out-of-fold predictions.
            test_predictions (np.ndarray): Test set predictions.
            train_df (pd.DataFrame): Training DataFrame.
        """
        # Save predictor and feature importance using pickle
        model_info = {
            'predictor': predictor,
            'feature_importance': feature_importance
        }
        with open('optimized_car_price_model.pkl', 'wb') as f:
            pickle.dump(model_info, f)
        logging.info("Trained model and feature importance saved to 'optimized_car_price_model.pkl'.")
        
        # Create and save submission file
        submission = pd.DataFrame({
            'Id': range(len(test_predictions)),
            'Predicted': np.round(test_predictions).astype(int)
        })
        submission_file = 'optimized_car_price_predictions.csv'
        submission.to_csv(submission_file, index=False)
        logging.info(f"Predictions saved to '{submission_file}'.")
        
        # Save feature importance
        importance_file = 'feature_importance.csv'
        feature_importance.to_csv(importance_file)
        logging.info(f"Feature importance saved to '{importance_file}'.")
        
        # Save model performance metrics
        oof_metrics = pd.DataFrame({
            'Metric': ['RMSE', 'R2'],
            'Value': [
                np.sqrt(mean_squared_error(train_df['price'], oof_predictions)),
                r2_score(train_df['price'], oof_predictions)
            ]
        })
        metrics_file = 'model_performance_metrics.csv'
        oof_metrics.to_csv(metrics_file, index=False)
        logging.info(f"Model performance metrics saved to '{metrics_file}'.")

# Unit Test to Ensure Missing Columns are Handled Correctly
def test_create_features_missing_columns():
    """
    Unit test to verify that the create_features method handles missing 'rare_and_exotic' correctly.
    """
    predictor = OptimizedCarPricePredictor()
    
    # Sample data without 'rare_and_exotic'
    sample_data = pd.DataFrame({
        'power': [100, 150],
        'curb_weight': [2000, 2500],
        'engine_cap': [2.0, 3.0],
        'depreciation': [500, 700],
        'vehicle_age': [2, 3],
        'vehicle_type_luxury_sedan': [1, 0],
        'premium_ad_car': [1, 1]
        # 'rare_and_exotic' is intentionally missing
    })
    
    # Initialize feature names
    cleaned_data = predictor.initialize_feature_names(sample_data)
    
    # Create features
    processed_data = predictor.create_features(cleaned_data)
    
    # Assert 'rare_and_exotic' exists
    assert 'rare_and_exotic' in processed_data.columns, "Missing 'rare_and_exotic' column after feature creation."
    assert all(processed_data['rare_and_exotic'] == 0), "'rare_and_exotic' should be filled with 0."
    
    print("Test passed: Missing 'rare_and_exotic' handled correctly.")

# Uncomment the following line to run the unit test
# test_create_features_missing_columns()

# Main Execution
if __name__ == "__main__":
    # Read data
    try:
        train_df = pd.read_csv("preprocessing/2024-10-21-silan/train_cleaned.csv")
        test_df = pd.read_csv("preprocessing/2024-10-21-silan/test_cleaned.csv")
        logging.info("Data files loaded successfully.")
    except FileNotFoundError as e:
        logging.error(f"Data file not found: {e}")
        raise
    
    # Initialize and train the predictor
    predictor = OptimizedCarPricePredictor(n_folds=5)
    oof_predictions, test_predictions, feature_importance = predictor.fit_predict(train_df, test_df)
    
    # Save model and results
    predictor.save_model_and_results(predictor, feature_importance, oof_predictions, test_predictions, train_df)
    
    # Output prediction statistics
    logging.info("\nPrediction Statistics:")
    logging.info(f"Minimum predicted price: {test_predictions.min():,.0f}")
    logging.info(f"Maximum predicted price: {test_predictions.max():,.0f}")
    logging.info(f"Mean predicted price: {test_predictions.mean():,.0f}")
    logging.info(f"Median predicted price: {np.median(test_predictions):,.0f}")