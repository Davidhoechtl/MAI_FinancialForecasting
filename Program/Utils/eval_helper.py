import numpy as np
import pandas as pd

def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, verbose: bool = True):
    """
    Evaluate binary classification performance (0/1 labels).

    Computes confusion matrix, accuracy, precision, recall, and F1 score.

    :param y_true: Ground truth labels (array-like of 0/1)
    :param y_pred: Predicted labels (array-like of 0/1)
    :param verbose: If True, prints results; otherwise returns dict
    :return: dict with metrics and confusion matrix
    """

    # Ensure NumPy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Confusion matrix elements
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    confusion_matrix = np.array([[tn, fp],
                                 [fn, tp]])

    # Metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan

    # Assemble results
    results = {
        "confusion_matrix": confusion_matrix,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    # Optional printout
    if verbose:
        print("Confusion Matrix:")
        print(pd.DataFrame(confusion_matrix,
                           columns=["Predicted Down (0)", "Predicted Up (1)"],
                           index=["Actual Down (0)", "Actual Up (1)"]))
        print()
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")

    return results