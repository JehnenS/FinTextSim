import pandas as pd
import numpy as np
from tqdm import tqdm

class MLDatasetBuilder:
    """
    Class to build ML dataset based on CSV predictors and target df
    Include text features from the ML models
    Ensure that we have the same observations by merging on target base
    """
    def __init__(self, 
                 target_df: pd.DataFrame, # dataframe with year, ticker and target
                 financials: pd.DataFrame, # dataframe where financial features are stored
                 text_features_fts: pd.DataFrame = None, # transformer-based sentence-level aggregated text features --> ticker and year_of_report as identifier
                 text_features_am: pd.DataFrame = None,
                 text_features_mpnet: pd.DataFrame = None,
                 text_features_distil_roberta: pd.DataFrame = None,
                 text_features_lda_tf: pd.DataFrame = None,
                 text_features_lda_tfidf: pd.DataFrame = None,
                 text_features_nmf_tf: pd.DataFrame = None,
                 text_features_nmf_tfidf: pd.DataFrame = None
                ):
        
        self.target_df = target_df
        self.financials = financials
        self.text_features_fts = text_features_fts
        self.text_features_am = text_features_am
        self.text_features_mpnet = text_features_mpnet
        self.text_features_distil_roberta = text_features_distil_roberta
        self.text_features_lda_tf = text_features_lda_tf
        self.text_features_nmf_tf = text_features_nmf_tf
        self.text_features_lda_tfidf = text_features_lda_tfidf
        self.text_features_nmf_tfidf = text_features_nmf_tfidf

        #print("Duplicates in financials:", self.financials.duplicated(subset=["ticker", "year"]).sum())
        #print("Duplicates in text_features_lda_tf:", self.text_features_lda_tf.duplicated(subset=["ticker", "year_of_report"]).sum())
        #print("Duplicates in target:", self.target_df.duplicated(subset=["ticker", "year"]).sum())



    def build_target_base(self):
        """
        Build the base for building the single dataframes --> grounded on the same observations

        CHANGE:
        The base now aligns each *target year t+1* with available *predictor-year t* 
        (e.g., 2024 EPS change is predicted using 2023 financials/text data).
        """
        #Step 1: Start base with all target rows
        base = self.target_df[["ticker", "year"]].copy()

        #shift one year backward for alignment (predictors come from t, target from t+1)
        base["pred_year"] = base["year"] - 1

        #Step 2: Restrict base if text features exist --> only include cases with available text data
        if self.text_features_fts is not None:
            base = base.merge(
                self.text_features_fts[["ticker", "year_of_report"]],
                left_on=["ticker", "pred_year"],   #CHANGE: match target t+1 with text year t
                right_on=["ticker", "year_of_report"], 
                how="inner"
            )
            base = base[["ticker", "year", "pred_year"]]



        # Step 3: Restrict target to base
        target_base = self.target_df.merge(base, on=["ticker", "year"], how="inner")

        print(f"Target base shape: {target_base.shape}")
        return target_base


    def _merge_text_financials(self, target_base, financials, text_features = None, cols_to_drop = ["doc_id", "filing_date", "cik"]):
        """
        Helper method to merge text with financials
        """
        #merge financials and text with target base df
        if text_features is not None:
            #preparation of text features
            text_features = text_features.drop(columns = cols_to_drop, errors = "ignore").drop_duplicates(subset = ["ticker", "year_of_report"]) #drop unnecessary columns
            
            df_fin_text = (
                target_base.merge(
                    financials.rename(columns={"year": "pred_year"}), #merge on prediction year --> prediction year = year of report of the 10-K to predict t+1
                        on=["ticker", "pred_year"],
                        how="inner"
                    )
                    .merge(
                        text_features.rename(columns={"year_of_report": "pred_year"}),
                        on=["ticker", "pred_year"],
                        how="inner" #in P2, we have "left"?!
                    )
                )
        else:
            df_fin_text = None

        return df_fin_text


    def build_all(self):
        """
        Build consistent ML datasets:
            - Financials only
            - Financials + FTS
            - Financials + AM
            - Financials + MPNET
            - Financials + LDA
            - Financials + NMF
        
        CHANGE:
        Merges now align predictors (financials/text) from year t with the target from year t+1.
        """
        # Step 0: Create target base
        target_base = self.build_target_base()

        # Step 1: Financials only df --> merge base with financials on predictor year
        df_fin_only = target_base.merge(
            self.financials.rename(columns={"year": "pred_year"}),
            on=["ticker", "pred_year"],
            how="inner"
        )

        # Step 2: Financials + FTS / AM / MPNET / LDA / NMF
        df_fin_fts = self._merge_text_financials(target_base, self.financials, self.text_features_fts)
        df_fin_am = self._merge_text_financials(target_base, self.financials, self.text_features_am)
        df_fin_mpnet = self._merge_text_financials(target_base, self.financials, self.text_features_mpnet)
        df_fin_distil_roberta = self._merge_text_financials(target_base, self.financials, self.text_features_distil_roberta)
        
        df_fin_lda_tf = self._merge_text_financials(target_base, self.financials, self.text_features_lda_tf)
        df_fin_lda_tfidf = self._merge_text_financials(target_base, self.financials, self.text_features_lda_tfidf)
        df_fin_nmf_tf = self._merge_text_financials(target_base, self.financials, self.text_features_nmf_tf)
        df_fin_nmf_tfidf = self._merge_text_financials(target_base, self.financials, self.text_features_nmf_tfidf)
        

        # Step 3: Print dataset shapes for sanity checking
        print(f"df_fin_only: {df_fin_only.shape}")
        print(f"df_fin_fts: {df_fin_fts.shape}")
        print(f"df_fin_am: {df_fin_am.shape}")
        print(f"df_fin_mpnet: {df_fin_mpnet.shape}")
        print(f"df_fin_distil_roberta: {df_fin_distil_roberta.shape}")
        #print(f"df_fin_lda_tf: {df_fin_lda_tf.shape}")
        print(f"df_fin_lda_tfidf: {df_fin_lda_tfidf.shape}")
        #print(f"df_fin_nmf_tf: {df_fin_nmf_tf.shape}")
        print(f"df_fin_nmf_tfidf: {df_fin_nmf_tfidf.shape}")

        #assert df_fin_only.shape[0] == df_fin_fts.shape[0] == df_fin_am.shape[0] == df_fin_mpnet.shape[0] == df_fin_lda_tf.shape[0] == df_fin_lda_tfidf.shape[0] == df_fin_nmf_tf.shape[0] == df_fin_nmf_tfidf.shape[0], "Mismatch between number of rows of the dfs"
        

        return df_fin_only, df_fin_fts, df_fin_am, df_fin_mpnet, df_fin_distil_roberta, df_fin_lda_tf, df_fin_lda_tfidf, df_fin_nmf_tf, df_fin_nmf_tfidf
