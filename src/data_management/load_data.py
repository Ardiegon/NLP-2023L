import pandas as pd
import numpy as np

from os.path import join
from configs.paths import DATA_DOUBLEQUALITY, DATA_NLP2023, DATA_DIR

MAX_COMMENT_LEN = 300

def load_raw_data_from_path(path:str) -> pd.DataFrame: 

    columns_of_interest = {
        DATA_DOUBLEQUALITY: ["content", "doubleQuality"],
        DATA_NLP2023: ["salusBcContent.description", "salusBcContent.doubleQuality"]
    }

    if path not in columns_of_interest.keys():
        raise RuntimeError("Given path is not handled in this method, use pd.read_csv and pick columns by hand, or add it here.")
    
    df = pd.read_csv(path)
    columns = columns_of_interest[path]

    if path==DATA_DOUBLEQUALITY:
        df = df[(df['language']=="POL")&(df['content'].str.len() < MAX_COMMENT_LEN)][columns]
        df = df.rename(columns={"content":"comment","doubleQuality":"label"})


    elif path ==DATA_NLP2023:
        df = df[(df['salusBcContent.language']=="DEU")&(df['salusBcContent.description'].str.len() < MAX_COMMENT_LEN)][columns] 
        df = df.rename(columns={"salusBcContent.description":"comment","salusBcContent.doubleQuality":"label"})
        
    df["comment"] = df["comment"].str.replace("\r", "")
    df["comment"] = df["comment"].str.replace("\n", "")
    df["comment"] = df["comment"].str.replace("\)", "")
    df["comment"] = df["comment"].str.replace("\(", "")
    df["comment"] = df["comment"].str.replace("-", "")
    df["comment"] = df["comment"].str.replace(":", "")

    df = df.reset_index().drop(columns=["index"])
    return df

def load_dataset(language:str):
    df_positive = pd.read_csv(join(DATA_DIR, f"{language}_positive.csv"))
    df_negative = pd.read_csv(join(DATA_DIR, f"{language}_negative.csv"))
    df = pd.concat([df_positive, df_negative], ignore_index=True).drop(columns=["Unnamed: 0"])
    return df

def load_dataset_encoded(language:str):
    df = pd.read_csv(join(DATA_DIR, f"{language}_encoded.csv")).drop(columns=["Unnamed: 0"])
    df["encoded"] = df["encoded"].apply(eval).apply(np.array)
    return df

if __name__ == "__main__":
    df = load_dataset("polish")
    print(df[df["label"]==True].head())
    print(df[df["label"]==False].head())

