import numpy as np
import torch


def calculate_confusion_matrix(prediction: np.array, target: np.array, n_classes=4, device="cuda") -> np.array:
    """
    Calculate confusion matrix.

    NAMGYU: TODO - remove device and replace torch with numpy (or convert back to numpy for the final output)

    Args:
        prediction (np.array): 2d-array of shape (n_samples, n_points_per_sample) with categorical values
        target (np.array): 2d-array of shape (n_samples, n_points_per_sample) with categorical values
        n_classes: total number of classes
        device (str): Device to run the computation on, either "cpu" or "cuda".

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

    prediction = torch.from_numpy(prediction).to(device)
    target = torch.from_numpy(target).to(device)

    confusion_matrices = torch.zeros((n_samples, n_classes, n_classes), device=prediction.device, dtype=torch.int64)

    # Create a 2D grid of indices (B*N, 2) indices, where each index is a pair of (target, output)
    indices = torch.stack([target, prediction], dim=-1)
    indices = indices.view(-1, 2)

    # Convert the 2D indices into linear indices (0, 0) -> 0, (0, 1) -> 1, (0, 2) -> 2, (0, 3) -> 3, (1, 0) -> 4, ...
    increment_values = confusion_matrices.new_ones(n_samples * n_points)
    linear_indices = indices[:, 0] * n_classes + indices[:, 1]

    # Flatten the output tensor
    output_tensor_flat = confusion_matrices.view(n_samples, -1)

    # Add 1 to the confusion matrix at the linear indices 
    output_tensor_flat.scatter_add_(1, linear_indices.view(n_samples, n_points),
                                    increment_values.view(n_samples, n_points))

    return confusion_matrices.cpu().numpy()
