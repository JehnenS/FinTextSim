import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt



class FinTargetExtractor:
    """
    Class to extract the label for the direction change of financial variables, such as earnings
    Possible improvements:
        Filter_0_na_rows (see utils_ml and utils_feature_creation from Paper 2)
    
    Cover multiple ranges: FY, Q1, Q2, Q3
    """
    def __init__(self, 
                 result_dict: dict, #path to result dict
                 yearly_cat: str = "financials",
                 quarterly_cat:str = "quarterly_financials",
                 target_table_name: str = "income_growth", #table in which the target variable can be found
                 target_variable_name: str = "growthEPSDiluted", #name of the target variable
                 target_period:str = "FY", #define the target period --> FY, Q1, Q2, Q3, Q4
                 min_year:int = 1990,
                 max_year:int = 2030,
                 kpi_min_abs_value: float = 0.01, 
                 kpi_max_abs_value: float = 5.0,
                 window:int = 4, #window to normalize changes of the variable
                 adjust_variable:bool = True, #boolean for normalization of target over window periods
                 binary_label:bool = True #boolean for transformation of label into binary classes
                ):
        
        self.result_dict = result_dict
        self.yearly_cat = yearly_cat
        self.quarterly_cat = quarterly_cat
        self.target_table_name = target_table_name
        self.target_variable_name = target_variable_name
        self.target_period = target_period
        
        self.min_year = min_year
        self.max_year = max_year
        self.kpi_min_abs_value = kpi_min_abs_value
        self.kpi_max_abs_value = kpi_max_abs_value
        self.window = window

        self.adjust_variable = adjust_variable
        self.binary_label = binary_label

        

    def extract_target_variable(self, compute_growth = False):
        """
        Function extract the target variable from the corresponding data
        result_dict: result_dict loaded by load_result_dict
        """
        print(f"----------Target variable: {self.target_variable_name}------------\n")

        
        #create list to store results per row --> transform into df later on
        rows = []

        #iterate over each entry in result dictionary --> grouped by ticker
        for ticker, data in tqdm(self.result_dict.items(), desc = "Progress"):
            if self.target_period == "FY":
                target_input = data.get(self.yearly_cat).get(self.target_table_name, [])  ##get the correct input from which we extract the target variable
            else:
                target_input = data.get(self.quarterly_cat).get(self.target_table_name, [])  ##get the correct input from which we extract the target variable

            for entry in target_input: #iterate over each entry in target input
                year = int(entry.get("calendarYear")) #--> corresponds to the fiscal year: e.g. AAPL: 'date': '2024-12-28','symbol': 'AAPL', calendarYear': '2025',  'period': 'Q1', --> Q1 of fiscal year 2025 even though quarter end date lies in 2024
                period = entry.get("period") #--> ensure that we look at the desired period
                target_variable = entry.get(self.target_variable_name) #extract the target variable

                #ensure that we have a year, valid target variable and that we extract the relevant target period
                if year and target_variable is not None and period == self.target_period:
                    rows.append({
                        "ticker": ticker,
                        "year": year,
                        "target": target_variable
                    })
        
        #transform list of dictionaries into pandas dataframe
        target_df = pd.DataFrame(rows).sort_values(["ticker", "year"]).reset_index(drop=True)
        target_df.drop_duplicates(subset = ["ticker", "year"], inplace = True) #drop duplicates for safety

        #handle cases where we want the growth rate instead of absolute value (if growth rate is not already the target)
        if compute_growth:
            print("Compute growth of target")
            target_df["target"] = target_df.groupby(["ticker"])["target"].pct_change() #overwrite the absolute target with growth rate

        return target_df

    def adjust_by_moving_average(self, target_df):
        """
        Subtract the average of the last `window` years from the current year's target.
        Operates per ticker (firm).
        If we extract quarter-based targets, we normalize them by the exact same quarter: e.g. Q1 2024 is normalized by Q1 2023, Q1 2022, Q1 2021, Q1 2020 --> cover seasonalities

        Similar to Chen et al. 2022
        """
        print("Adjust target by moving average")
        target_df = target_df.copy()
        
        #sort to ensure proper rolling computation
        target_df = target_df.sort_values(["ticker", "year"])
    
        #compute rolling mean for last window years
        target_df["mean_past"] = (
            target_df
            .groupby("ticker")["target"]
            .transform(lambda x: x.shift(1).rolling(window = self.window, min_periods = self.window - 1).mean())
        )
    
        #subtract the mean of the past x years from current value
        target_df["target"] = target_df["target"] - target_df["mean_past"]

        target_df = target_df[abs(target_df["mean_past"]).notna()].reset_index(drop = True) #drop rows where the mean_past is NA --> observations where no rolling window could be constructed
    
        #drop helper column
        target_df = target_df.drop(columns=["mean_past"])
    
        print(f"Adjusted target using {self.window}-year rolling average.")

        
        return target_df


    def filter_target_df(self, target_df):
        """
        Filter target df by years and kpi values
        """
        target_df = target_df.copy()
        print(f"Observations: {target_df.shape[0]}")
    
        #always filter by year
        filtered_target_df = target_df[
            (target_df["year"] >= self.min_year) &
            (target_df["year"] <= self.max_year)
        ].reset_index(drop=True)
    
        print(f"Observations after year filtering: {filtered_target_df.shape[0]}")
    
        #optional max absolute value filter
        if self.kpi_max_abs_value is not None:
            filtered_target_df = filtered_target_df[
                abs(filtered_target_df["target"]) < self.kpi_max_abs_value
            ].reset_index(drop=True)
            print(f"After applying abs. max value < {self.kpi_max_abs_value}: {filtered_target_df.shape[0]}")
    
        #optional min absolute value filter
        if self.kpi_min_abs_value is not None:
            filtered_target_df = filtered_target_df[
                abs(filtered_target_df["target"]) >= self.kpi_min_abs_value
            ].reset_index(drop=True)
            print(f"After applying abs. min value >= {self.kpi_min_abs_value}: {filtered_target_df.shape[0]}")
    
        return filtered_target_df

    def binary_transform_target_variable(self, target_df):
        """
        Transform the target variable to binary format --> 0 for decrease, 1 for increase
        """
        target_df["target"] = target_df["target"].apply(lambda x: 1 if x > 0 else 0)
        return target_df


    def plot_target_distribution(self, target_df):
        fig, axes = plt.subplots(1,2, figsize = (16,4))
        axes[0].boxplot(target_df["target"])
        axes[0].set_title(f"Boxplot of {self.target_variable_name} values")
        axes[0].grid(alpha = 0.2)
        
        axes[1].hist(target_df["target"], edgecolor = "white")
        axes[1].set_title(f"Histogram of {self.target_variable_name} values")
        axes[1].grid(alpha = 0.2)
        
        plt.tight_layout()
        plt.show()

    def get_target_df(self, compute_growth = False):
        """
        Wrapper method to get the target df
        """

        df = self.extract_target_variable(compute_growth = compute_growth)
        if self.adjust_variable: #filter after removing the outliers
            df = self.adjust_by_moving_average(df)
        df = self.filter_target_df(df)
        
        if self.binary_label:
            df = self.binary_transform_target_variable(df)

        print("Targets extracted.\n")
        return df