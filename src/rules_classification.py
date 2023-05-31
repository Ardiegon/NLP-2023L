import pandas as pd
import json
import os
import sys
#from model_utils import plot_results
from data_management.load_data import load_dataset


if len(sys.argv) < 4:
    print("Too few arguments!")
    sys.exit(1)
else:
    if sys.argv[3] == "train":
        if len(sys.argv) < 7:
            print("Too few arguments for training!")
            sys.exit(1)

        df = pd.read_csv(sys.argv[4], delimiter=sys.argv[7])
        # print(df)
        # print(df.columns)
        content_column, label_column = [sys.argv[5], sys.argv[6]]
        df = df[[content_column, label_column]]

    elif sys.argv[3] == "test":
        if len(sys.argv) == 4:
            print("Nothing to test!")
            sys.exit(1)
        data = sys.argv
    else:
        print("Unknown mode!")
        sys.exit(1)

workdir = os.getcwd()
df = load_dataset("polish")
print(df)

# with open(sys.argv[2]) as json_file:
#     rules_settings = json.load(json_file)

# rules = rules_settings["rules"][sys.argv[1]]
# langs = rules_settings["rules"]
# if sys.argv[1] not in langs:
#     print("This language is not supported!")
#     sys.exit(1)


# df[content_column] = df[content_column].apply(str.lower)
# for ch in ['.', ',', '"', '"']:
#     df[content_column] = df[content_column].str.replace(ch, '')
# df['prediction'] = 0


# for rule in rules:
#     df.loc[:, 'prediction'] = [True if pred == True else True if rule in com else False for com,
#                                pred in zip(df[content_column], df['prediction'])]


# print(df['a'], df['prediction'])
# # print(df[df[label_column] == df['prediction']].shape[0] / df.shape[0])


# if sys.argv[3] == "train":
#     #print(df[df[label_column] == df['prediction']].shape[0] / df.shape[0])
#     plot_results(df['a'], df['prediction'])
# else:
#     pass
