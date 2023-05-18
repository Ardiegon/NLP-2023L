import pandas as pd

from configs.paths import DATA_DOUBLEQUALITY, DATA_NLP2023

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

if __name__ == "__main__":
    df = load_raw_data_from_path(DATA_NLP2023)
    pd.set_option('display.max_columns', None)  
    print(df.shape)
    print(repr(df["comment"][94]))
