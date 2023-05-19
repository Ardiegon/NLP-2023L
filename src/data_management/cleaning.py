import numpy as np
import nltk
import pandas as pd

from enum import Enum
from os.path import join
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

from typing import List

from configs.paths import DATA_DOUBLEQUALITY, DATA_NLP2023, DATA_DIR, POLISH_STOPWORDS
from data_management.load_data import load_dataset 

class ModelNames(Enum):
    POLISH = "Voicelab/sbert-base-cased-pl"
    GERMAN = "T-Systems-onsite/cross-en-de-roberta-sentence-transformer"
    MULTILINGUAL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

def encode(sentences:"pd.Series[str]", language:ModelNames)->"pd.Series[list[float]]":
    model = SentenceTransformer(language.value)
    embeddings = model.encode(sentences)
    return pd.Series(embeddings.tolist())

def get_stopwords(language:str):
    if language == "polish":
        with open(POLISH_STOPWORDS, 'r', encoding='utf-8') as file:
            custom_stopwords = [word.strip() for word in file.readlines()]
        return set(custom_stopwords)
    if language == "german":
        return set(stopwords.words(language))
    
def remove_stopwords(sentences:"pd.Series[str]", language:str)->"pd.Series[str]":
    stop_words = get_stopwords(language)
    def _remove_stopwords(text:str):
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        return ' '.join(filtered_tokens)
    return sentences.apply(_remove_stopwords)

def main():
    # nltk.download('stopwords')
    # nltk.download('punkt')
    lang = "polish"
    df = load_dataset(lang)
    df["comment"] = remove_stopwords(df["comment"], "polish")
    df["encoded"] = encode(df["comment"], ModelNames.POLISH)
    df.to_csv(join(DATA_DIR, f"{lang}_encoded.csv"))

if __name__ == "__main__":
    main()

