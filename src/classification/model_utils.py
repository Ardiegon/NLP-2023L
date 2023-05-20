import numpy as np
import matplotlib.pyplot as plt

from joblib import dump, load
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score


def save_model(path, model):
    dump(model, path)

def load_model(path):
    return load(path) 

def calc_results(ground_truth, predicted):
    accuracy = accuracy_score(ground_truth, predicted)
    precision = precision_score(ground_truth, predicted)
    recall = recall_score(ground_truth, predicted)
    f1 = f1_score(ground_truth, predicted)
    return accuracy, precision, recall, f1

def plot_results(ground_truth, predicted, title = "Confusion Matrix", save_path = ""):
    matrix = confusion_matrix(ground_truth, predicted)
    
    accuracy = accuracy_score(ground_truth, predicted)
    precision = precision_score(ground_truth, predicted)
    recall = recall_score(ground_truth, predicted)
    f1 = f1_score(ground_truth, predicted)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(4, 6))
    
    ax1.imshow(matrix, interpolation='nearest', cmap=plt.cm.Greens)
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
    
    ax2.text(0.5, 0.5, f'Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}',
             ha='center', va='center', fontsize=12)
    ax2.axis('off')
    
    fig.tight_layout(h_pad=2)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

