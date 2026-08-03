import os

import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from load_data import load_data
matplotlib.use("Agg")

#DATA_PATH = r"E:\2-1\ML\placement.csv"


CHARTS_DIR=os.path.join(os.path.dirname(__file__),"static","charts")

def _chart_path(filename:str)-> str:
    os.makedirs(CHARTS_DIR,exist_ok=True)
    return os.path.join(CHARTS_DIR,filename)
def _save(filename:str):
    plt.tight_layout()
    plt.savefig(_chart_path(filename), bbox_inches="tight")
    plt.close("all")
def run_eda()-> dict:
    data=load_data()
    charts =[]
    missing=data.isnull().sum()
    missing_pct =(missing/len(data))*100
    missing_df=pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    missing_df=missing_df[missing_df["missing_count"] >0].sort_values(
        "missing_count",ascending=False
    )

    if not missing_df.empty:
      plt.figure(figsize=(10, 5))
      sns.barplot(x=missing_df.index, y=missing_df["missing_pct"])
      plt.xticks(rotation=45, ha="right")
      plt.ylabel("Missing %")
      plt.title("Missing Values by column")
      _save("missing_values.png")
      charts.append("missing_values.png")

    duplicate_count=int(data.duplicated().sum())

    target_counts =data["PlacementsStatus"].value_counts().to_dict()
    plt.figure()
    sns.countplot(x="PlacementsStatus", data=data)
    plt.xlabel("Placement Status (0 =Not Placed. 1=Placed")
    plt.ylabel("Count")
    plt.title("Count of Placement Status")
    _save("target_distribution.png")
    charts.append("target_distribution.png")

    hist_cols =[
        "CGPA","AttendancePercent","AptitudeTestScore",
        "SoftSkillRating"
    ]
"""sns.set_style("whitegrid")   # fixed here
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

def show(title=""):
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    plt.close("all")

# 1. LOAD Data
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("File not found")
data = pd.read_csv(DATA_PATH)
print("="*80)
print("1.DATA Loaded")
print("="*80)
print("shape:", data.shape)
print("\nfirst 5 rows:\n", data.head())

#2. Basic INFO ?STructure
print("\n" + "=" *80)
print("2.BASIC INFO")
print("="*80)
print(data.info())
print("\nColumn dtypes:\n",data.dtypes)
print("\nDescribe (numeric):\n", data.describe())
print("\nDescribe (categorical):\n", data.describe(include="object"))
"""
#3.Missing values
print("\n"+ "=" * 80)
print("3. MISSING VALUES")
print("=" * 80)
missing =data.isnull().sum()
missing_pct =(missing/len(data))*100
missing_df =pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
missing_df=missing_df[missing_df["missing_count"] >0].sort_values("missing_count",ascending=False)
print(missing_df)


"""if not missing_df.empty:
    plt.figure(figsize = (10,5),dpi=100)
    sns.barplot(x=missing_df.index,y=missing_df["missing_pct"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Missing %")
    plt.title("Missing Values")
    show()
    plt.figure(figsize = (12,6),dpi=100)
    sns.heatmap(data.isnull(), cbar=False, cmap="virdis")
    plt.title("Missing Vlaue Heatmap")
    show() """
#4.DUPLICATEs
print("\n" + "=" * 80)
print("4. Duplicate Values")
print("=" * 80)
print("Duplicate rows:",data.duplicated().sum())

#5.TARGET VARIABLE distribution
print("\n" + "=" * 80)
print("5. TARGET VARIABLE _PlacementStatus")
print("=" * 80)
print(data["PlacementStatus"].value_counts())

plt.figure(dpi=125)
sns.countplot(x="PlacementStatus", data=data)
plt.xlabel("Placement Status (0 =Not Placed. 1=Placed")
plt.ylabel("Count")
plt.title("Count of Placement Status")
plt.show()

#6.NUMERIC FEATURE DISTRIBUTIONS
print("\n" +"="* 80)

if __name__ =="__main__":
    results = run_eda()
    print(results)