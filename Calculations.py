import numpy as np
import scipy
from sklearn.metrics import roc_auc_score, roc_curve


def calculate_eer_and_dprime(labels, scores):
    """
    Calculate Equal Error Rate (EER) and d-prime.
    
    Parameters:
        labels (list): True labels (1 for genuine, 0 for impostor).
        scores (list): Model scores for each pair.
    
    Returns:
        float: EER value.
        float: d-prime value.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]

    # Calculate d-prime using AUC
    auc_value = roc_auc_score(labels, scores)
    d_prime = np.sqrt(2) * scipy.special.erfinv(2 * auc_value - 1)
    
    return eer, d_prime

# Example usage
labels = [1, 1, 0, 0, 1]  # True labels (genuine/impostor)
scores = [0.9, 0.8, 0.4, 0.2, 0.7]  # Scores from the classifier
eer, d_prime = calculate_eer_and_dprime(labels, scores)
print(f"EER: {eer}, d-prime: {d_prime}")
