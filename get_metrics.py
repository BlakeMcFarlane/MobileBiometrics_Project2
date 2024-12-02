from sklearn.metrics import roc_curve, auc, DetCurveDisplay, confusion_matrix as sk_confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def get_roc(gen_scores, imp_scores):
    # Combine genuine and impostor scores and create labels
    scores = np.concatenate([gen_scores, imp_scores])
    labels = np.concatenate([np.ones(len(gen_scores)), np.zeros(len(imp_scores))])

    # Compute ROC curve and ROC area
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='red', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Random guess line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    return

def get_det(gen_scores, imp_scores):
    scores = np.concatenate([gen_scores, imp_scores])
    labels = np.concatenate([np.ones(len(gen_scores)), np.zeros(len(imp_scores))])

    DetCurveDisplay.from_predictions(labels, scores)
    plt.title('DET Curve', fontsize=14)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Plot confusion matrix using true labels, predicted labels, and class names.

    Parameters:
    - y_true: array-like of shape (n_samples,)
        True labels.
    - y_pred: array-like of shape (n_samples,)
        Predicted labels.
    - classes: array-like of shape (n_classes,)
        Class labels (names).

    Returns:
    - None
    """
    # Compute confusion matrix
    cm = sk_confusion_matrix(y_true, y_pred, labels=classes)

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
    plt.title('Confusion Matrix', fontsize=14)
    plt.show()


def compute_eer(gen_scores, imp_scores):
    # Combine scores and labels
    scores = np.concatenate([gen_scores, imp_scores])
    labels = np.concatenate([np.ones(len(gen_scores)), np.zeros(len(imp_scores))])

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Compute EER
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    print(f'Equal Error Rate (EER): {eer * 100:.2f}% at threshold {eer_threshold:.4f}')
    return eer, eer_threshold

def compute_authentication_accuracy(gen_scores, imp_scores, threshold):
    # Make authentication decisions
    gen_decisions = gen_scores >= threshold  # True if accepted
    imp_decisions = imp_scores < threshold   # True if correctly rejected

    # Compute total correct decisions
    correct_gen = np.sum(gen_decisions)
    correct_imp = np.sum(imp_decisions)
    total_attempts = len(gen_scores) + len(imp_scores)
    total_correct = correct_gen + correct_imp

    # Compute accuracy
    authentication_accuracy = total_correct / total_attempts * 100
    print(f'Authentication Accuracy: {authentication_accuracy:.2f}%')
    return authentication_accuracy
