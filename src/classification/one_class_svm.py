# Script for classification using SVM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm

from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, HalvingGridSearchCV
from data_management.load_data import load_dataset_encoded

SEED = 12345

def calc_results(ground_truth, predicted):
    precision = precision_score(ground_truth, predicted)
    recall = recall_score(ground_truth, predicted)
    f1 = f1_score(ground_truth, predicted)
    return precision, recall, f1

def plot_results(ground_truth, predicted, title = "Confusion Matrix", save_path = ""):
    matrix = confusion_matrix(ground_truth, predicted)
    
    precision = precision_score(ground_truth, predicted)
    recall = recall_score(ground_truth, predicted)
    f1 = f1_score(ground_truth, predicted)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(4, 6))
    
    ax1.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.set_title(title)
    tick_marks = np.arange(len(np.unique(ground_truth)))
    ax1.set_xticks(tick_marks)
    ax1.set_yticks(tick_marks)
    ax1.set_xticklabels(np.unique(ground_truth))
    ax1.set_yticklabels(np.unique(ground_truth))
    ax1.set_xticklabels(['Positive', 'Negative'])
    ax1.set_yticklabels(['Positive', 'Negative'], rotation = "vertical")
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    thresh = matrix.max() / 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax1.text(j, i, format(matrix[i, j], 'd'),
                     horizontalalignment='center',
                     color='white' if matrix[i, j] > thresh else 'black')
    
    ax1.tick_params(axis='both', which='both', length=0)
    
    ax2.text(0.5, 0.5, f'Precision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}',
             ha='center', va='center', fontsize=12)
    ax2.axis('off')
    
    fig.tight_layout(h_pad=2)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def get_train_test_dataset(language:str)->tuple:
    df = load_dataset_encoded(language)
    train, test_negative = train_test_split(df[df["label"]==False], test_size=0.06, random_state=SEED)
    test_positive = df[df["label"]==True]
    return train, pd.concat([test_negative, test_positive], ignore_index=True).sample(frac=1, random_state=SEED)


def main():
    train, test = get_train_test_dataset("polish")
    
    train_encodings = np.vstack(train["encoded"].to_numpy())
    test_encodings = np.vstack(test["encoded"].to_numpy())
    test_answers = test["label"].apply(lambda x: 1 if not x else -1).tolist()
    
    nu = 0.3
    gamma = 0.00001
    model = svm.OneClassSVM(nu=nu, kernel="rbf", gamma=gamma)

    model.fit(train_encodings)
    test_result = model.predict(test_encodings)
    
    plot_results(test_answers, test_result, title=f"Predicted with SVM.\nIs comment about double quality?", save_path="svm_results.png")
    

if __name__ == "__main__":
    main()
