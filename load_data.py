import os
import pandas as pd

DATA_PATH = r"C:\Users\david\PycharmProjects\Project_(ML_2)\placement.csv"

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return df

def get_data_summary() -> dict:
    df = load_data()
    summary = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_counts": {col: int(df[col].isnull().sum()) for col in df.columns},
        "preview": df.head(10).to_dict("records"),
    }
    return summary

if __name__ == "__main__":
    print(get_data_summary())
