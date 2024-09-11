import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize


def plot_roc_curves(y_true, y_pred, lw=1):
    """
    Plot ROC curves based on the shape of y_pred.

    Args:
    y_true: Array of ground truth labels.
    y_pred: Array of predicted probabilities; shape may indicate binary or multi-class.
    """
    # Determine if y_pred is for binary or multi-class classification
    if y_pred.ndim == 1:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='blue', lw=lw, label=f'ROC curve (AUC = {roc_auc:.2f})')
    else:
        # Multi-class classification
        # Binarize the labels for multi-class ROC
        classes = np.unique(y_true)
        y_true_bin = label_binarize(y_true, classes=classes)
        n_classes = y_true_bin.shape[1]

        # Compute ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=lw,
                     label=f'Class {classes[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)  # Diagonal line for random guess
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.show()


def compute_classification_metrics(y_true, y_pred, threshold=0.5):
    """
    Computes important classification metrics for binary classification.

    Parameters:
    - y_true: Ground truth binary labels (0 or 1).
    - y_pred: Predicted scores (between 0 and 1).
    - threshold: Threshold for converting predicted scores to binary labels (default is 0.5).

    Returns:
    A dictionary with the following metrics:
    - accuracy
    - precision
    - recall
    - f1_score
    - auc
    """
    # Convert predicted scores to binary predictions based on the threshold
    y_pred_binary = (y_pred >= threshold).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    auc = roc_auc_score(y_true, y_pred)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc
    }
