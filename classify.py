import argparse
import numpy as np
import pandas as pd

from configs.classify_config import *
from data_management.cleaning import encode, remove_stopwords, filter_weird_characters
from classification.model_utils import load_model

def get_options(language:str, model:str)->dict:
    map_lang_to_opts = {
        "polish"+"svm": opts_polish_svm,
        "german"+"svm": opts_german_svm,
        "polish"+"bayes": opts_polish_bayes,
        "german"+"bayes": opts_german_bayes
    }
    try:
        return map_lang_to_opts[language+model]
    except KeyError:
        print("Sorry, such combination of model and language is not implemented.")
        exit(1)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("comments", nargs="+", type=str, help="Comments that you want to check for double quality.")
    parser.add_argument("--model", default = "bayes", type=str, help="Model type.")
    parser.add_argument("--language", default = "polish", type=str, help="Language in which comments are written.")
    return parser.parse_args()

def main(args):
    opts = get_options(args.language, args.model)

    model = load_model(opts["model"])

    how_many_comments = len(args.comments)
    assert how_many_comments > 0

    data = pd.Series(args.comments)
    data = filter_weird_characters(data)
    data = remove_stopwords(data, opts["language"])
    data = encode(data, opts["language"])

    input_encodings = np.vstack(data.to_numpy())
    predictions = model.predict(input_encodings)

    for i in range(how_many_comments):
        print(f"\"{args.comments[i]}\" predicted as: {'positive' if predictions[i]==-1 else 'negative'}")

if __name__ == "__main__":
    args = parse_args()
    main(args)