# Script for classification using Naive Bayes

import numpy as np
import pandas as pd

from sklearn.naive_bayes import ComplementNB
from classification.model_utils import save_model, plot_results, get_test_sizes
from data_management.load_data import load_dataset_encoded
from sklearn.model_selection import train_test_split

from configs.paths import BAYES_GERMAN, BAYES_POLISH
from configs.general import LANGUAGE

SEED = 999
BAYES_NON_NEGATIVE_CONST = 100
USE_AUGMENTED_DATA = True


def get_train_test_dataset(language:str)->tuple:
    df = load_dataset_encoded(language, augumented=USE_AUGMENTED_DATA)
    train_negative, test_negative = train_test_split(df[df["label"]==False], test_size=get_test_sizes(USE_AUGMENTED_DATA)[0], random_state=SEED)
    train_positive, test_positive = train_test_split(df[df["label"]==True], test_size=get_test_sizes(USE_AUGMENTED_DATA)[1], random_state=SEED)
    return pd.concat([train_negative, train_positive], ignore_index=True).sample(frac=1, random_state=SEED), pd.concat([test_negative, test_positive], ignore_index=True).sample(frac=1, random_state=SEED)

def main():
    lang = LANGUAGE
    map_lang_to_path = {
        "polish": BAYES_POLISH,
        "german": BAYES_GERMAN
    }

    train, test = get_train_test_dataset(lang)
    
    train_encodings = np.vstack(train["encoded"].to_numpy()) + BAYES_NON_NEGATIVE_CONST
    train_answers = train["label"].apply(lambda x: 1 if not x else -1).tolist()
    test_encodings = np.vstack(test["encoded"].to_numpy()) + BAYES_NON_NEGATIVE_CONST
    test_answers = test["label"].apply(lambda x: 1 if not x else -1).tolist()
    
    model = ComplementNB(force_alpha=True)

    model.fit(train_encodings, train_answers)
    test_result = model.predict(test_encodings)
    
    plot_results(test_answers, test_result, title=f"Predicted with Naive Bayes.\nIs comment about double quality?", save_path=f"results\\bayes_results_{lang}_augumentation_{USE_AUGMENTED_DATA}.png")
    save_model(map_lang_to_path[lang], model)

if __name__ == "__main__":
    main()
