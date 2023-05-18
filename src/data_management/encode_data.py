import numpy as np
import pandas as pd

from enum import Enum
from sentence_transformers import SentenceTransformer

from configs.paths import DATA_DOUBLEQUALITY, DATA_NLP2023

class ModelNames(Enum):
    POLISH = "Voicelab/sbert-base-cased-pl"
    GERMAN = "T-Systems-onsite/cross-en-de-roberta-sentence-transformer"
    MULTILINGUAL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

def remove_stopwords():
    pass

def encode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    pass

def cosine(u:list, v:list) -> pd.DataFrame:
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def test_encoding():
    test_sentences = [
        "Krowa lubi jeść trawę",
        "Rolnik kosi trawę przy krowach",
        "Praca czyni człowiekiem sukcesu",
        "Harte Arbeit und Ausdauer sind der Schlüssel zum Erfolg",
        "Das Leben ist eine Reise, genieße jeden Moment.",
        "Die Zeit heilt alle Wunden, sei geduldig.",
        "Erfolg ist kein Glück, sondern harte Arbeit.",
        "Życie to podróż, ciesz się każdą chwilą.",
        "Czas leczy wszystkie rany, bądź cierpliwy.",
        "Sukces to nie kwestia szczęścia, lecz ciężkiej pracy.",
        "Kot leży na łące"
    ]

    model = SentenceTransformer(ModelNames.POLISH.value)
    embeddings = model.encode(test_sentences)

    for val, e in enumerate(embeddings[1:]):
        sim = cosine(embeddings[0], e)
        print("Sentence = ", test_sentences[val+1], "; similarity = ", sim)

if __name__ == "__main__":
    test_encoding()

