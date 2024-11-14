from dotenv import load_dotenv
import os
load_dotenv()

from data.rmse_calculator import calculate_rmse
from final_report.approach1.without_cluster.models.catboost import train_catboost_models
from final_report.approach1.without_cluster.models.lightgbm import train_lightgbm_models
from final_report.approach1.without_cluster.models.gradientboost import train_gradient_boosting_models
from final_report.approach1.without_cluster.models.xgboost import train_xgboost_models

# def train_lightgbm_models(train_file_path, test_file_path, prediction_output_path):



# FINAL_REPORT_SUBMISSION_PATH="final_report\final_report_submission.csv"
# FINAL_REPORT_TRAIN_DATA="final_report\preprocessing\self_experiment\train_cleaned.csv"
# FINAL_REPORT_TEST_DATA="final_report\preprocessing\self_experiment\test_cleaned.csv"

# FINAL_REPORT_APPROACH_one_SUBMISSION_PATH="final_report\approach1\final_report_submission.csv"
# FINAL_REPORT_APPROACH_two_SUBMISSION_PATH="final_report\approach2\final_report_submission.csv"


def main():
    print('Training model...')
    
# FINAL_REPORT_APPROACH_one_WITHOUT_cluster_LightGBM_WEIGHT_PATH="final_report\approach1\without_cluster\lightgbm_models.pkl"
# FINAL_REPORT_APPROACH_one_WITHOUT_cluster_XGBoost_WEIGHT_PATH="final_report\approach1\without_cluster\xgboost_models.pkl"
# FINAL_REPORT_APPROACH_one_WITHOUT_cluster_CATBOOST_WEIGHT_PATH="final_report\approach1\without_cluster\catboost_models.pkl"
# FINAL_REPORT_APPROACH_one_WITHOUT_cluster_GRADIENT_BOOSTING_WEIGHT_PATH="final_report\approach1\without_cluster\gradient_boosting_models.pkl"

    train_xgboost_models(
        os.getenv('FINAL_REPORT_TRAIN_DATA'),
        os.getenv('FINAL_REPORT_TEST_DATA'),
        os.getenv('FINAL_REPORT_APPROACH_one_SUBMISSION_PATH'),
        os.getenv('FINAL_REPORT_APPROACH_one_WITHOUT_cluster_XGBoost_WEIGHT_PATH')        
    )
    # train_lightgbm_models(
    #     os.getenv('FINAL_REPORT_TRAIN_DATA'),
    #     os.getenv('FINAL_REPORT_TEST_DATA'),
    #     os.getenv('FINAL_REPORT_APPROACH_one_SUBMISSION_PATH'),
    #     os.getenv('FINAL_REPORT_APPROACH_one_WITHOUT_cluster_LightGBM_WEIGHT_PATH')
    # )
    # train_catboost_models(
    #     os.getenv('FINAL_REPORT_TRAIN_DATA'),
    #     os.getenv('FINAL_REPORT_TEST_DATA'),
    #     os.getenv('FINAL_REPORT_APPROACH_one_SUBMISSION_PATH'),
    #     os.getenv('FINAL_REPORT_APPROACH_one_WITHOUT_cluster_CATBOOST_WEIGHT_PATH')
    # )
    # train_gradient_boosting_models(
    #     os.getenv('FINAL_REPORT_TRAIN_DATA'),
    #     os.getenv('FINAL_REPORT_TEST_DATA'),
    #     os.getenv('FINAL_REPORT_APPROACH_one_SUBMISSION_PATH'),
    #     os.getenv('FINAL_REPORT_APPROACH_one_WITHOUT_cluster_GRADIENT_BOOSTING_WEIGHT_PATH')
    # )
    print('Training model done.')
    print('evalute...')
    RSME = calculate_rmse(os.getenv('FINAL_REPORT_APPROACH_one_SUBMISSION_PATH'))
    print(f'RSME: {RSME}')


def switch_main(model_name):
    """
    Train and evaluate the model based on the given model name.
    'xgboost': XGBoost
    'lightgbm': LightGBM
    'catboost': CatBoost
    'gradientboost': GradientBoosting
    """
    print('Training model...')
    
    if model_name == 'xgboost':
        train_xgboost_models(
            os.getenv('FINAL_REPORT_TRAIN_DATA'),
            os.getenv('FINAL_REPORT_TEST_DATA'),
            os.getenv('FINAL_REPORT_APPROACH_one_SUBMISSION_PATH'),
            os.getenv('FINAL_REPORT_APPROACH_one_WITHOUT_cluster_XGBoost_WEIGHT_PATH')        
        )
    elif model_name == 'lightgbm':
        train_lightgbm_models(
            os.getenv('FINAL_REPORT_TRAIN_DATA'),
            os.getenv('FINAL_REPORT_TEST_DATA'),
            os.getenv('FINAL_REPORT_APPROACH_one_SUBMISSION_PATH'),
            os.getenv('FINAL_REPORT_APPROACH_one_WITHOUT_cluster_LightGBM_WEIGHT_PATH')
        )
    elif model_name == 'catboost':
        train_catboost_models(
            os.getenv('FINAL_REPORT_TRAIN_DATA'),
            os.getenv('FINAL_REPORT_TEST_DATA'),
            os.getenv('FINAL_REPORT_APPROACH_one_SUBMISSION_PATH'),
            os.getenv('FINAL_REPORT_APPROACH_one_WITHOUT_cluster_CATBOOST_WEIGHT_PATH')
        )
    elif model_name == 'gradientboost':
        train_gradient_boosting_models(
            os.getenv('FINAL_REPORT_TRAIN_DATA'),
            os.getenv('FINAL_REPORT_TEST_DATA'),
            os.getenv('FINAL_REPORT_APPROACH_one_SUBMISSION_PATH'),
            os.getenv('FINAL_REPORT_APPROACH_one_WITHOUT_cluster_GRADIENT_BOOSTING_WEIGHT_PATH')
        )
    else:
        print('Invalid model name.')
        return
    print('Training model done.')
    print('evalute...')
    RSME = calculate_rmse(os.getenv('FINAL_REPORT_APPROACH_one_SUBMISSION_PATH'))
    print(f'RSME: {RSME}')