import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

class StockTargetExtractor:
    """
    Class to extract the label for stock price movements from pre_days before filing date to post_days past filing date
    10-K filing events
    """
    def __init__(self):
        """
        For CAR calculation, most of the time literature uses a three day window centered around filing date / earnings call, etc. (e.g. Huang2023)
        """


    def extract_event_windows(self,
                              stock_df, 
                              filing_df, 
                              pre_days:int = 1,
                              post_days:int = 1,
                              max_gap_pre:int = 5,
                              max_gap_post:int = 5
                              ):
        """
        function to extract relevant information from stock df based on pre_days (trading days) before filing date and post_days (trading days) past filing date

        medium-term price momentum: past year (252 trading days) up until one month before filing date (252/12) trading days --> 252 pre_days, -252/12 post_days
        short-term price reversal: past month before filing date (252/12) pre_days, 0 post_days
        """
        #convert date to datetime
        stock_df['date'] = pd.to_datetime(stock_df['date'])
        filing_df['filing_date'] = pd.to_datetime(filing_df['filing_date'])
    
        #sort df
        stock_df = stock_df.sort_values(['ticker', 'date']).copy()
    
        #create dictionary for faster lookup per ticker
        stock_by_ticker = {ticker: df for ticker, df in stock_df.groupby('ticker')}
    
        #merge dataframes to only iterate over filing dates which are also in stock info
        filing_dates_merge = filing_df.merge(stock_df, left_on = ["ticker", "filing_date"], right_on = ["ticker", "date"], how = "inner")
        filing_dates_merge = filing_dates_merge[[col for col in filing_dates_merge.columns if col in filing_df.columns]] #ensure that we maintain only the original columns
    
        #create list to store results
        all_windows = []
    
        #iterate over each row in merged df
        for _, row in tqdm(filing_dates_merge.iterrows(), total = filing_dates_merge.shape[0], desc="Extracting event windows"):
            ticker = row['ticker']
            filing_date = row['filing_date']
            year = row['year'] #year from period of report originating from employee - count - filing date API
    
            if ticker not in stock_by_ticker:
                continue
    
            #extract the data relevant for ticker from dictionary
            ticker_data = stock_by_ticker[ticker]
            ticker_data = ticker_data.reset_index(drop=True)
    
            #find index of filing_date in trading days
            date_idx = ticker_data[ticker_data['date'] == filing_date].index
            if len(date_idx) == 0:
                continue
            idx = date_idx[0]
    
            #define start and end idx
            start_idx = max(0, idx - pre_days) #handle cases where there are less than pre_days observations before filing date
            end_idx = min(len(ticker_data), idx + post_days + 1) #handle cases where there are less than post_days + 1 observations after filing date

            #filter for "valid" event windows --> e.g. if start/end date are significantly different from filing date due to clipping in stock_df, we cannot use this window            
            window_df = ticker_data.iloc[start_idx:end_idx].copy() #extract the relevant data from full set

            #sanity filter - we need consistent data around the filing date - Only keep rows within a reasonable date range
            window_df = window_df[
                (window_df['date'] >= filing_date - pd.Timedelta(days = pre_days + max_gap_pre)) &
                (window_df['date'] <= filing_date + pd.Timedelta(days = post_days + max_gap_post))
            ]
            #handle cases where we do not have enough filing date informations
            if len(window_df) < (pre_days + post_days + 1): 
                continue
            else:
                window_df['event_filing_date'] = filing_date #add the filing date to df
                window_df['ticker'] = ticker #add ticker to df
                window_df['year'] = year #add the filing date to df
    
                all_windows.append(window_df)
    
        event_window_df = pd.concat(all_windows, ignore_index=True)
    
        return event_window_df

    def calculate_targets(self, event_window_df, rm, rf, ticker_beta_mapping):
        """
        Calculate target metrics for the event windows
        - relative movement
        - CAR based on CAPM
        - CAR based on Fama-French
        """
        df = event_window_df.copy()

        #merge market return and risk free rate
        all_dates = pd.date_range(df['date'].min(), df['date'].max()) #set whole date range
        rm = rm.set_index('date').reindex(all_dates).ffill().reset_index().rename(columns={'index':'date'}) #forward-fill missing values for rm and rf
        rf = rf.set_index('date').reindex(all_dates).ffill().reset_index().rename(columns={'index':'date'})
        df = df.merge(rm, on = "date", how = "left") #merge rm and rf with df
        df = df.merge(rf, on = "date", how = "left")

        #add beta for each ticker
        df["beta"] = df["ticker"].map(ticker_beta_mapping)
        df = df[df["beta"].notna()] #filter out rows without beta

        missing_rf = df['risk_free_rate_daily'].isna().sum()
        missing_mkt = df['market_return'].isna().sum()
        print(f"Missing RF: {missing_rf}, Missing Market Return: {missing_mkt}")
        

        #add expected CAPM return
        df["expected_return_CAPM"] = df["risk_free_rate_daily"] + df["beta"] * (df["market_return"] - df["risk_free_rate_daily"])
        df["abnormal_return_CAPM"] = df["daily_return"] - df["expected_return_CAPM"]

        grouped = df.groupby(['ticker', 'event_filing_date', 'year'])

        def safe_rel_move(x):
            """
            Safe calculation of relative stock movement
            """
            if x.iloc[0] == 0:
                return np.nan  # return NA if last price is equal to 0
            return (x.iloc[-1] - x.iloc[0]) / x.iloc[0]

        targets = grouped.agg(
            rel_move=('adjClose', lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0]),
            CAR_CAPM=('abnormal_return_CAPM', 'sum'),
            volume = ('volume', 'sum'),
            mean_adjClose = ('adjClose', 'mean'),
            beta = ('beta', 'mean')
        ).reset_index()

        return targets

    def filter_targets(self, targets,
                       min_volume=1e4,
                       min_price=1.0,
                       beta_bounds=(-5, 5),
                       rel_move_quantiles=(0.01, 0.99),
                       car_capm_quantiles=(0.01, 0.99)):
        """
        Filters out irrational or low-quality event targets based on:
          - low trading volume
          - penny stocks
          - extreme beta values
          - extreme relative moves or CAR_CAPM values
        """
        df = targets.copy()
    
        #basic volume and price filters
        df = df[(df['volume'] >= min_volume) & (df['mean_adjClose'] >= min_price)]
    
        #beta sanity filter - absolute values + abs(beta) > 0 --> expected relation to market
        df = df[(df['beta'] >= beta_bounds[0]) & (df['beta'] <= beta_bounds[1]) & (df['beta'].abs() > 0)]
    
        #winsorize or filter extreme rel_move and CAR_CAPM values
        rel_low, rel_high = df['rel_move'].quantile(rel_move_quantiles)
        car_low, car_high = df['CAR_CAPM'].quantile(car_capm_quantiles)
        df = df[
            (df['rel_move'].between(rel_low, rel_high)) &
            (df['CAR_CAPM'].between(car_low, car_high))
        ]
    
        print(f"Filtered targets: {len(df)} remaining from {len(targets)}")
        return df
        

    def classify_targets(self, targets, target_name, rel_cols = ["ticker", "year", "target"], abs_target_threshold = 0.0005):
        """
        Generate binary labels from targets
        further filter for targets exceeding absolute value --> cleaning the signal and remove noise from targets which are close to 0+-x
        """
        targets[f"{target_name}_abs"] = targets[target_name].abs() #get absolute value of target
        print(f"\nNumber of observations before filtering by absolute target value: {targets.shape[0]}")
        targets_filtered = targets[targets[f"{target_name}_abs"] >= abs_target_threshold] #filter based on threshold
        print(f"Number of observations after filtering by absolute target value {abs_target_threshold}: {targets_filtered.shape[0]}")

        #binary classification label
        targets_filtered["target"] = (targets_filtered[target_name] > 0).astype(int)

        targets_filtered = targets_filtered[rel_cols]
        return targets_filtered