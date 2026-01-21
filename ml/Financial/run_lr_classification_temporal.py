import pandas as pd
import numpy as np
import cupy as cp
from tqdm import tqdm
import os
import pickle
import cudf


os.chdir("/mnt/sdb1/home/simonj") #set working directory

#import json
#import argparse
#parser = argparse.ArgumentParser()
#parser.add_argument("--config", required=True, help="Path to config JSON file")
#args = parser.parse_args()

#with open(args.config, "r") as f:
 #   config = json.load(f)

# Now you can access config["target_variable"], etc.
#print(f"Running {__name__} with config:", config)



results_dict = {}


config = {
    "target_period": "FY",
    "kpi_min_abs_value": 0.0,
    "kpi_max_abs_value": 1e9,
    "binary_label": True,
    "target_variable": "returnOnAssets",
    "target_table_name": "ratios",  
    "compute_growth": True,
    "exclude_quarter_features": True,
    "fintextsim": "htl_temporal",
    "adjust_target_variable": True,
    "feature_set": "swade",
    "oversample_method": None,
    "test_year_start": 2023,
    "min_year_target": 2010,
    "max_year_target": 2030,
    "n_bootstrap_auc": 10000
}

results_dict["config"] = config

#print(f"\nSample: {results_dict["config"].get("sample")}")
#print(f"Outlier detection method: {results_dict["config"].get("outlier_detection")}")
print(f"Oversampling: {results_dict["config"].get("oversample_method")}")
print(f"Start of test-period: {results_dict["config"].get("test_year_start")}")
#print(f"Feature Set: {results_dict["config"].get("feature_set")}")
print(f"Exclude quarter features: {results_dict["config"].get("exclude_quarter_features")}")
print(f"Target variable: {results_dict["config"].get("target_variable")} - {results_dict["config"].get("target_period")}")
print(f"Computation of growth for target variable: {results_dict["config"].get("compute_growth")}")
print(f"Adjustment of target variable: {results_dict["config"].get("adjust_target_variable")}\n")

from ml.utils_ml import load_result_loc, load_data_loc, load_sentiment_loc

result_loc = load_result_loc(results_dict, model_name = "lr", base_path = "paper1/Results/ML/Financials_temporal")

#-----------Load data
data_loc = load_data_loc(results_dict)

with open(data_loc, "rb") as file:
    data = pickle.load(file)

data.keys()

results = data["results"]
results_economic = results["economic"] #extract economic data to make it cleanly ticker-based
results.pop("economic", None) #remove economic from results to make it cleanly ticker-based

#----------get target
from feature_creation.FinTargetExtractor import FinTargetExtractor


ft_extractor = FinTargetExtractor(
    result_dict = results,
    target_table_name = results_dict["config"].get("target_table_name"), #table in which the target variable can be found
    target_variable_name = results_dict["config"].get("target_variable"), #name of the target variable
    target_period = results_dict["config"].get("target_period"), #can be either FY, Q1, Q2, Q3, Q4
    min_year = results_dict["config"].get("min_year_target"), #year of reported values --> in df: min_year -1 as we need the 2011 predictors to predict 2012 results --> shift by -1 so that 2012 results align with 2011 predictors for merging
    max_year = results_dict["config"].get("max_year_target"),
    kpi_min_abs_value = results_dict["config"].get("kpi_min_abs_value"), 
    kpi_max_abs_value = results_dict["config"].get("kpi_max_abs_value"),
    binary_label = results_dict["config"].get("binary_label"), #boolean for transformation of label into binary classes
    adjust_variable = results_dict["config"].get("adjust_target_variable")
)

target_df = ft_extractor.get_target_df(results_dict["config"].get("compute_growth"))



#---------Load features
#define the paths
loc_fin = "paper1/Data/Features/fin_features.csv"
loc_swade = "paper1/Data/Features/swade_features.csv"

fin_features = pd.read_csv(loc_fin)
swade_features = pd.read_csv(loc_swade)


#decide if we want to include quarter-based features
if results_dict["config"].get("exclude_quarter_features"):
    quarter_suffixes = ("_Q1", "_Q2", "_Q3", "_Q4")
    
    quarter_based_features = [
        col for col in fin_features.columns
        if col.endswith(quarter_suffixes)
    ]
    print(f"Number of quarter-based features: {len(quarter_based_features)}")
    fin_features = fin_features[[col for col in fin_features.columns if col not in quarter_based_features]]
    print(f"Shape of df without quarter-based features: {fin_features.shape}")


#assign financiancials based on feature set
if results_dict["config"].get("feature_set") == "swade":
    financials = swade_features
else:
    financials = fin_features


loc_fts = os.path.join("paper1/Data/Features/text_features", f"text_features_bertopic_{config.get("fintextsim")}.csv")
loc_am = "paper1/Data/Features/text_features/text_features_am_masked.csv"
loc_mpnet = "paper1/Data/Features/text_features/text_features_mpnet_masked.csv"
loc_distil_roberta = "paper1/Data/Features/text_features/text_features_distil_roberta_masked.csv"
loc_lda_tf = "paper1/Data/Features/text_features/text_features_lda_tf.csv"
loc_lda_tfidf = "paper1/Data/Features/text_features/text_features_lda_tfidf.csv"
loc_nmf_tf = "paper1/Data/Features/text_features/text_features_nmf_tf.csv"
loc_nmf_tfidf = "paper1/Data/Features/text_features/text_features_nmf_tfidf.csv"


doc_vectors_fts = pd.read_csv(loc_fts)
doc_vectors_am = pd.read_csv(loc_am)
doc_vectors_mpnet = pd.read_csv(loc_mpnet)
doc_vectors_distil_roberta  = pd.read_csv(loc_distil_roberta)
#doc_vectors_lda_tf = pd.read_csv(loc_lda_tf)
doc_vectors_lda_tfidf = pd.read_csv(loc_lda_tfidf)
#doc_vectors_nmf_tf = pd.read_csv(loc_nmf_tf)
doc_vectors_nmf_tfidf = pd.read_csv(loc_nmf_tfidf)

#--------------------LLM text features
from ml.utils_ml import models


#----create stock features
from ml.MLDatasetBuilderCSV import MLDatasetBuilder
from ml.utils_ml import categorical_cols, cols_to_exclude, text_only_cols
from ml.PrepML import PrepML


    
results_dict.setdefault("results", {})              # Ensure 'results' exists


    

#generate ML datasets for different textual feature sets
builder = MLDatasetBuilder(
    target_df = target_df,
    financials = financials,
    text_features_fts = doc_vectors_fts,
    text_features_am = doc_vectors_am,
    text_features_mpnet = doc_vectors_mpnet,
    text_features_distil_roberta = doc_vectors_distil_roberta,
    #text_features_lda_tf = doc_vectors_lda_tf,
    text_features_lda_tfidf = doc_vectors_lda_tfidf,
    #text_features_nmf_tf = doc_vectors_nmf_tf,
    text_features_nmf_tfidf = doc_vectors_nmf_tfidf
)

df_fin_only, df_fin_fts, df_fin_am, df_fin_mpnet, df_fin_distil_roberta, df_fin_lda_tf, df_fin_lda_tfidf, df_fin_nmf_tf, df_fin_nmf_tfidf = builder.build_all()

dfs = [df_fin_only, df_fin_fts, df_fin_am, df_fin_mpnet, df_fin_distil_roberta, df_fin_lda_tf, df_fin_lda_tfidf, df_fin_nmf_tf, df_fin_nmf_tfidf] #set df_fin_text as base for text only df --> drop same rows based on financial features; then filter columns in X_train and X_test for sentiment features
df_names = ["fin", "fin+fts", "fin+am", "fin+mpnet", "fin+dr", "fin+lda+tf", "fin+lda+tfidf", "fin+nmf+tf", "fin+nmf+tfidf"]  
        
#clean the dfs if a df is None
df_names = [name for df, name in zip(dfs, df_names) if df is not None]
dfs = [df for df in dfs if df is not None]


for i, df in enumerate(dfs):
    print(f"\n-------------DF: {df_names[i]}---------------------")
    mlprep = PrepML(
        df = df,
        target_name = "target"
    )
    
    X_train, X_test, y_train, y_test, neg_instances, pos_instances, X, y = mlprep.run_lr_preprocessing(
        categorical_cols = categorical_cols,
        cols_to_exclude = cols_to_exclude,
        threshold_columns = 0.5,
        threshold_rows = 0.4,
        test_start_year = results_dict["config"].get("test_year_start"),
        oversample_method = results_dict["config"].get("oversample_method")
    )
        
    #--------train and evaluate
    from ml.Classifier import Classifier
        
    classifier = Classifier(
        X_train, X_test, y_train, y_test, neg_instances, pos_instances, X, y
    )
        
    y_train_pred, y_test_pred, y_train_prob, y_test_prob = classifier.lr_classification()
        
    #evaluate
    print("\nTrain Set:")
    train_results, train_results_filtered = classifier.evaluate_model(
        y_true = y_train,
        y_pred = y_train_pred,
        y_prob = y_train_prob,
        n_bootstrap_auc = 0
    )
        
    print("\nTest Set:")
    test_results, test_results_filtered = classifier.evaluate_model(
        y_true = y_test,
        y_pred = y_test_pred,
        y_prob = y_test_prob,
        n_bootstrap_auc = results_dict["config"].get("n_bootstrap_auc")
    )

    print("\nk-fold cross validation:")
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])

    cv_results = classifier.evaluate_cv_lr(X_full, y_full, n_splits=5)
    print(cv_results)
  
        
    results_dict["results"][df_names[i]] = {
        'train_results': train_results,
        'train_results_filtered': train_results_filtered,
        "test_results": test_results, 
        "test_results_filtered": test_results_filtered,
        #"model": model,
        #"shap_values": shap_values,
        #"explainer": explainer,
        #"shap_interaction_values": shap_interaction_values,
        "X_train": X_train,
        "X_test": X_test,
        "feature_names": list(X.columns),
        "cv_results": cv_results
    }


with open(result_loc, "wb") as file:
    pickle.dump(results_dict, file)
print(f"Data saved to {result_loc}")