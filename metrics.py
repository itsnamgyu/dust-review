import numpy as np
import pandas as pd


import numpy as np
import torch

def calculate_confusion_matrix(output: np.array, target: np.array, n_category=4, device='cuda'):
    """
    Calculate confusion matrix
    :param output: np.array of shape (B, N)
    :param target: np.array of shape (B, N)
    :param device: 'cpu' or 'cuda' to specify where to run the computation
    :return: torch.Tensor of shape (B, n_category, n_category)
        p0 p1 p2 p3
    t0  9  5  1  0
    t1  8  2  3  4
    t2  0  1  2  3
    t3  1  2  3  0

    """

    B, N = output.shape

    output = torch.from_numpy(output).to(device)
    target = torch.from_numpy(target).to(device)

    confusion_matrices = torch.zeros((B, n_category, n_category), device=output.device, dtype=torch.int64)

    # Create a 2D grid of indices (B*N, 2) indices, where each index is a pair of (target, output)
    indices = torch.stack([target, output], dim=-1)
    indices = indices.view(-1, 2)

    # Convert the 2D indices into linear indices (0, 0) -> 0, (0, 1) -> 1, (0, 2) -> 2, (0, 3) -> 3, (1, 0) -> 4, ...
    increment_values = torch.ones((B * N,), device=device, dtype=confusion_matrices.dtype)
    linear_indices = indices[:, 0] * n_category + indices[:, 1]

    # Flatten the output tensor
    output_tensor_flat = confusion_matrices.view(B, -1)

    # Add 1 to the confusion matrix at the linear indices 
    output_tensor_flat.scatter_add_(1, linear_indices.view(B, N), increment_values.view(B, N))

    return confusion_matrices.detach().cpu().numpy()
