from dotenv import load_dotenv
import os
load_dotenv()

from data.rmse_calculator import calculate_rmse
from final_report.approach1.without_cluster.models.lightgbm import train_lightgbm_models


# def train_lightgbm_models(train_file_path, test_file_path, prediction_output_path):



# FINAL_REPORT_SUBMISSION_PATH="final_report\final_report_submission.csv"
# FINAL_REPORT_TRAIN_DATA="final_report\preprocessing\self_experiment\train_cleaned.csv"
# FINAL_REPORT_TEST_DATA="final_report\preprocessing\self_experiment\test_cleaned.csv"

# FINAL_REPORT_APPROACH_one_SUBMISSION_PATH="final_report\approach1\final_report_submission.csv"
# FINAL_REPORT_APPROACH_two_SUBMISSION_PATH="final_report\approach2\final_report_submission.csv"


def main():
    print('Training model...')
    train_lightgbm_models(
        os.getenv('FINAL_REPORT_TRAIN_DATA'),
        os.getenv('FINAL_REPORT_TEST_DATA'),
        os.getenv('FINAL_REPORT_APPROACH_one_SUBMISSION_PATH')
    )
    print('Training model done.')
    # print('evalute...')
    # RSME = calculate_rmse(os.getenv('FINAL_REPORT_APPROACH_one_SUBMISSION_PATH'))
    # print(f'RSME: {RSME}')
