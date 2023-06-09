# Script for classification using SVM

import numpy as np
import pandas as pd

from sklearn import svm
from classification.model_utils import save_model, plot_results
from data_management.load_data import load_dataset_encoded
from sklearn.model_selection import train_test_split

from configs.paths import SVM_GERMAN, SVM_POLISH
from configs.general import LANGUAGE

SEED = 12345
USE_AUGMENTED_DATA = True


def get_train_test_dataset(language:str)->tuple:
    df = load_dataset_encoded(language, augumented=USE_AUGMENTED_DATA)
    train, test_negative = train_test_split(df[df["label"]==False], test_size=0.063, random_state=SEED)
    test_positive = df[df["label"]==True][:63]
    return train, pd.concat([test_negative, test_positive], ignore_index=True).sample(frac=1, random_state=SEED)

def main():
    lang = LANGUAGE
    map_lang_to_path = {
        "polish": SVM_POLISH,
        "german": SVM_GERMAN
    }

    train, test = get_train_test_dataset(lang)
    
    train_encodings = np.vstack(train["encoded"].to_numpy())
    test_encodings = np.vstack(test["encoded"].to_numpy())
    test_answers = test["label"].apply(lambda x: 1 if not x else -1).tolist()
    
    nu = 0.3
    gamma = 0.00001
    model = svm.OneClassSVM(nu=nu, kernel="rbf", gamma=gamma)

    model.fit(train_encodings)
    test_result = model.predict(test_encodings)
    
    plot_results(test_answers, test_result, title=f"Predicted with SVM.\nIs comment about double quality?", save_path=f"results\\svm_results_{lang}_augumented_{USE_AUGMENTED_DATA}.png")
    save_model(map_lang_to_path[lang], model)

if __name__ == "__main__":
    main()
