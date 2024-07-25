import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from typing import Dict
import pandas as pd

def calculate_confusion_matrix(prediction: np.array, target: np.array, n_classes=4) -> np.array:
    """
    Calculate confusion matrix.

    Args:
        prediction (np.array): 2d-array of shape (n_samples, n_points_per_sample) with categorical values
        target (np.array): 2d-array of shape (n_samples, n_points_per_sample) with categorical values
        n_classes: total number of classes

    Returns:
        An array of shape (n_samples, n_classes, n_classes) representing the confusion matrix.

    Example:
        p0 p1 p2 p3
    t0  9  5  1  0
    t1  8  2  3  4
    t2  0  1  2  3
    t3  1  2  3  0
    """
    n_samples, n_points = prediction.shape

    # Initialize confusion_matrices
    confusion_matrices = np.zeros((n_samples, n_classes, n_classes), dtype=np.int64)

    # Create a 2D grid of indices (B*N, 2) indices, where each index is a pair of (target, output)
    indices = np.stack([target, prediction], axis=-1)
    indices = indices.reshape(n_samples, n_points, 2)

    # Convert the 2D indices into linear indices (0, 0) -> 0, (0, 1) -> 1, (0, 2) -> 2, (0, 3) -> 3, (1, 0) -> 4, ...
    increment_values = np.ones((n_samples, n_points), dtype=np.int64)
    linear_indices = indices[..., 0] * n_classes + indices[..., 1]

    # Flatten the output tensor
    output_tensor_flat = confusion_matrices.reshape(n_samples, -1)

    # Add 1 to the confusion matrix at the linear indices
    np.add.at(output_tensor_flat, (np.arange(n_samples)[:, None], linear_indices), increment_values)

    # Reshape the output tensor back to the original shape
    confusion_matrices = output_tensor_flat.reshape(n_samples, n_classes, n_classes)

    return confusion_matrices


def calculate_classification_metrics(cm: np.array, binary_classification_index: int = 2) -> Dict[str, float]:
    """
    Print classification metrics for confusion matrix. This is used by MetricTracker for metrics on regional and grid
    predictions.

    Args:
        cm: confusion matrix. ndarray of shape (n_samples, n_classes, n_classes)
        binary_classification_index: index of binary threshold for binary classification metrics
    
    Returns:
        A dictionary containing classification metrics.
    """

    bindex = binary_classification_index
    tn = np.sum(cm[:, :bindex, :bindex])
    fp = np.sum(cm[:, :bindex, bindex:])
    fn = np.sum(cm[:, bindex:, :bindex])
    tp = np.sum(cm[:, bindex:, bindex:])
    total = np.sum(cm)

    acc = np.sum(cm[:, 0, 0] + cm[:, 1, 1] + cm[:, 2, 2] + cm[:, 3, 3]) / total
    hard_acc = np.sum(cm[:, 2, 2] + cm[:, 3, 3]) / (fn + tp)
    far = fp / (fp + tp)
    pod = tp / (tp + fn)
    f1 = 2 * pod * (1 - far) / (pod + (1 - far))

    return {'acc': acc, 'hard_acc': hard_acc, 'far': far, 'pod': pod, 'f1': f1}


def calculate_regression_metrics(predictions: np.array, target: np.array) -> Dict[str, float]:
    """
    Calculate regression metrics.

    Args:
        prediction (np.array): 1d-array of shape (n_samples,) with continuous values
        target (np.array): 1d-array of shape (n_samples,) with continuous values

    Returns:
        A dictionary containing regression metrics.

    Example:
        nmb: 0.2, nme: 0.1, r: 0.8, rmse: 0.5
    """

    nmb = np.sum(predictions - target) / np.sum(target)
    nme = np.sum(np.abs(predictions - target)) / np.sum(target)
    r, _ = pearsonr(predictions, target)
    rmse = np.sqrt(mean_squared_error(target, predictions)) # np.sqrt(np.mean((predictions-target)**2))

    return {'nmb': nmb, 'nme': nme, 'r': r, 'rmse': rmse}

def convert_confusion_df_to_array(confusion_df: pd.DataFrame) -> np.array:
    """
    Convert confusion matrix dataframe to numpy array.

    Args:
        df: confusion matrix pandas dataframe

    Returns:
        A numpy array of shape (n_samples, n_classes, n_classes) representing the confusion matrix.
    """
    columns_to_convert = ['t0p0', 't0p1', 't0p2', 't0p3', 
                          't1p0', 't1p1', 't1p2', 't1p3', 
                          't2p0', 't2p1', 't2p2', 't2p3', 
                          't3p0', 't3p1', 't3p2', 't3p3']
    confusion_array = confusion_df[columns_to_convert].to_numpy().reshape(-1, 4, 4)

    return confusion_array