import pandas as pd
import json
import os
import sys
from classification.model_utils import plot_results
from data_management.load_data import load_dataset


def predict(df, rules, column_name):
    df['prediction'] = False
    for rule in rules:
        df.loc[:, 'prediction'] = [True if pred == True else True if rule in com else False for com,
                                   pred in zip(df[column_name], df['prediction'])]
    return df


if len(sys.argv) < 3:
    print("Too few arguments! You need to provide language and path to rules and, optionally, sentences to clasificate")
    sys.exit(1)

workdir = os.getcwd()

with open(sys.argv[2]) as json_file:
    rules_settings = json.load(json_file)

rules = rules_settings["rules"][sys.argv[1]]
langs = rules_settings["rules"]
if sys.argv[1] not in langs:
    print("This language is not supported!")
    sys.exit(1)


if len(sys.argv) == 3:
    df = predict(load_dataset(
        rules_settings["translate_lang"][sys.argv[1]]), rules, 'comment')
    plot_results(df['label'], df['prediction'])
else:
    sentences = sys.argv[3::]
    df = predict(pd.DataFrame(data=sentences, columns=[
                 'comment']), rules, 'comment')
    for com, pred in zip(df['comment'], df['prediction']):
        print(f"{com}: {pred}")
