import pandas as pd
import numpy as np
from pandas import to_datetime

def preprocess_data(path: str, test_size: float=0.8, split: bool=True) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = to_datetime(df["time"])
    df['timestamp'] = df['time'].astype(int) // 10**9
    df['rate'] = df.groupby('item')['rate'].transform(lambda x: x / x.max()) 
    df = df.sort_values(by="time")
    
    if not split:
        return df
    split = int(len(df) * test_size)
    train_df, test_df = df[:split], df[split:]
    return train_df, test_df

def preprocess_for_classification_report(test_df:pd.DataFrame) -> pd.DataFrame:
    if "is_buying" not in test_df.columns:
        test_df["is_buying"] = False
    test_df["is_buying"] = test_df["rate"] > 0
    return test_df