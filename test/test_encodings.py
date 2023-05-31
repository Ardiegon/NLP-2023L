import numpy as np
import pandas as pd
from data_management.cleaning import encode, ModelNames

from configs.general import LANGUAGE

def cosine_distance(u:list, v:list) -> pd.DataFrame:
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def test_encoding():
    test_sentences = pd.Series([
        "Harte Arbeit und Ausdauer sind der Schlüssel zum Erfolg",
        "Rolnik kosi trawę przy krowach",
        "Praca czyni człowiekiem sukcesu",
        "Das Leben ist eine Reise, genieße jeden Moment.",
        "Die Zeit heilt alle Wunden, sei geduldig.",
        "Erfolg ist kein Glück, sondern harte Arbeit.",
        "Życie to podróż, ciesz się każdą chwilą.",
        "Czas leczy wszystkie rany, bądź cierpliwy.",
        "Sukces to nie kwestia szczęścia, lecz ciężkiej pracy.",
        "Kot leży na łące",
        "Krowa lubi jeść trawę"
    ])

    embeddings = encode(test_sentences, LANGUAGE)
    print(type(embeddings))

    for val, e in enumerate(embeddings[1:]):
        sim = cosine_distance(embeddings[0], e)
        print("Sentence = ", test_sentences[val+1], "; similarity = ", sim)


if __name__ == "__main__":
    test_encoding()