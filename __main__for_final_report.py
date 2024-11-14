from final_report.approach1.train_and_evalute import switch_main as approach1_main
from final_report.approach1.without_cluster.error_analysis import main as approach1_without_cluster_error_analysis
# from final_report.approach1.cluster.cluster_base_on_error_train_and_evalute import main as approach1_cluster_main   
# from final_report.approach1.cluster.cluster_base_on_feature_train_and_error import main as approach1_cluster1_main
from final_report.approach2.sample_essemble import main as approach2_main
from final_report.approach3.sample_essemble_make_ref_price import main as approach3_main

if __name__ == "__main__":
    # """
    # Approach One    Train and evaluate the model based on the given model name.
    # 'xgboost': XGBoost
    # 'lightgbm': LightGBM
    # 'catboost': CatBoost
    # 'gradientboost': GradientBoosting
    # """
    # approach1_main('gradientboost')
    
    # """
    # Approach One Error Analysis
    # please train all models before running this function
    # """
    # approach1_without_cluster_error_analysis()
    
    
    # """
    # if you have intrested in clustering, you can run the following code
    # """
    # approach1_cluster_main() // cluster_base_on_error_train_and_evalute.py
    # approach1_cluster1_main() // cluster_base_on_feature_train_and_error.py
    
    # """
    # Approach Two Sample Ensemble
    # """
    # approach2_main()
    
    
    # """
    # Approach Three Sample Ensemble Make Ref Price
    # """
    approach3_main()