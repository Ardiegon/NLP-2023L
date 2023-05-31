import numpy as np
import pandas as pd
import logging
import nltk

from enum import Enum
from os.path import join
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

from configs.paths import DATA_DIR, POLISH_STOPWORDS
from data_management.load_data import load_dataset 

logging.disable(logging.INFO) 
logging.disable(logging.WARNING)

class ModelNames(Enum):
    POLISH = "Voicelab/sbert-base-cased-pl"
    GERMAN = "T-Systems-onsite/cross-en-de-roberta-sentence-transformer"
    MULTILINGUAL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

def encode(sentences:"pd.Series[str]", language:str)->"pd.Series[list[float]]":
    map_lang_to_name = {
        "polish": ModelNames.POLISH.value,
        "german": ModelNames.GERMAN.value,
        "multi": ModelNames.MULTILINGUAL.value
    }
    model = SentenceTransformer(map_lang_to_name[language])
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
    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = get_stopwords(language)
    def _remove_stopwords(text:str):
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        return ' '.join(filtered_tokens)
    return sentences.apply(_remove_stopwords)

def filter_weird_characters(sentences:"pd.Series[str]")->"pd.Series[str]":
    pattern = r'[^a-zA-Z0-9.,\!\?\sąćęłńóśźżĄĆĘŁŃÓŚŹŻäöüßÄÖÜ]'
    return sentences.str.replace(pattern, '', regex=True)


def main():
    lang = "german"
    df = load_dataset(lang)
    df["comment"] = filter_weird_characters(df["comment"])
    df["comment"] = remove_stopwords(df["comment"], lang)
    df["encoded"] = encode(df["comment"], lang)
    df.to_csv(join(DATA_DIR, f"{lang}_encoded_augm.csv"))

if __name__ == "__main__":
    main()

