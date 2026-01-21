import pandas as pd
from tqdm import tqdm

class CIKMainSymbolExtractor:
    def __init__(
        self, 
        result_dict,
        cat_name = "info",
        input_table_name = "information"
    ):
        self.result_dict = result_dict
        self.cat_name = cat_name
        self.input_table_name = input_table_name
        self.features = {}  # store dfs by table name
        
    def extract_symbols(self):
        """
        Extracts all symbols from CIK-based comapny information API --> mapping between all ciks from textual data and ticker symbols
        """
        # create list to store results per row (equals ticker-year combo)
        rows = []

        # iterate over each ticker
        for cik, data in tqdm(self.result_dict.items(), desc=f"Extracting from {self.input_table_name}"):
            table = data.get(self.cat_name).get(self.input_table_name, [])  # extract the relevant table

            # iterate over entries in the relevant table
            for entry in table:
                symbol = entry.get("symbol")
                if not symbol:  # catches None and empty strings
                    continue
            
                # skip if cik is missing
                if not cik:
                    continue
                
                row = {"cik": cik, "ticker": symbol}
                rows.append(row)

        df = pd.DataFrame(rows).sort_values(["ticker"]).reset_index(drop=True)
        print(f"Number of cik/tickers: {len(df)}")
        # drop rows where either ticker or cik is null
        df = df.dropna(subset=["cik", "ticker"])
        print(f"Number of cik/tickers after dropping na: {len(df)}")
        # remove duplicates
        df = df.drop_duplicates(subset=["cik", "ticker"]).reset_index(drop=True)
        print(f"Number of cik/tickers after dropping duplicates: {len(df)}")

        # check missing values
        print(df.isna().sum())
        
        # check if any cik or ticker is duplicated
        dup_ciks = df["cik"][df["cik"].duplicated()].unique()
        dup_tickers = df["ticker"][df["ticker"].duplicated()].unique()
        
        print("Duplicate CIKs:", dup_ciks)
        print("Duplicate tickers:", dup_tickers)

        return df