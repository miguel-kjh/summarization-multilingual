import torch

def rbf_kernel(X, sigma=1.0):
    """
    Calcula el kernel Gaussiano (RBF) para un tensor de entrada.
    Args:
        X (torch.Tensor): Tensor de forma (n_samples, n_features) o (batch_size, n_samples, n_features).
        sigma (float): Parámetro de dispersión del kernel.
    Returns:
        torch.Tensor: Matriz kernel de forma (batch_size, n_samples, n_samples).
    """
    if X.dim() == 2:
        X = X.unsqueeze(0)  # Agregar dimensión batch si no está presente

    pairwise_sq_dists = torch.cdist(X, X, p=2)**2  # Distancias cuadradas entre pares
    return torch.exp(-pairwise_sq_dists / (2 * sigma**2))

def center_kernel(K):
    """
    Centra la matriz kernel.
    Args:
        K (torch.Tensor): Matriz kernel original de forma (batch_size, n_samples, n_samples).
    Returns:
        torch.Tensor: Matriz kernel centrada.
    """
    batch_size, n, _ = K.size()
    ones = torch.ones((batch_size, n, n), device=K.device) / n
    return K - ones @ K - K @ ones + ones @ K @ ones

def hsic(X, Y, sigma=1.0):
    """
    Calcula el criterio de independencia Hilbert-Schmidt (HSIC) para un batch.
    Args:
        X (torch.Tensor): Tensor de forma (batch_size, n_samples, n_features).
        Y (torch.Tensor): Tensor de forma (batch_size, n_samples, m_features).
        sigma (float): Parámetro de dispersión del kernel.
    Returns:
        torch.Tensor: Tensor de valores HSIC de forma (batch_size,).
    """
    # Calcula los kernels Gaussianos para X e Y
    K_X = rbf_kernel(X, sigma)
    K_Y = rbf_kernel(Y, sigma)

    # Centra los kernels
    K_X_centered = center_kernel(K_X)
    K_Y_centered = center_kernel(K_Y)

    # Calcula HSIC para cada batch
    hsic_values = torch.einsum('bij,bij->b', K_X_centered, K_Y_centered) / (X.size(1)**2)
    return hsic_values

import torch.nn as nn

def pairwise_distances(X):
    """
    Calcula las distancias por pares para las muestras en el tensor X.
    Args:
        X (torch.Tensor): Tensor de forma (n_samples, n_features).
    Returns:
        torch.Tensor: Tensor de forma (n_samples, n_samples) con distancias por pares.
    """
    # Cálculo de distancias cuadradas: ||x_i - x_j||^2
    dist_sq = torch.cdist(X, X, p=2)**2
    return dist_sq

class LocalRelationLoss(nn.Module):
    def __init__(self):
        """
        Constructor para la función de pérdida basada en la conservación de relaciones locales.
        """
        super(LocalRelationLoss, self).__init__()

    def forward(self, X, Z):
        """
        Calcula la pérdida de conservación de relaciones locales.
        Args:
            X (torch.Tensor): Tensor de entrada original de forma (n_samples, n_features_original).
            Z (torch.Tensor): Tensor de salida proyectada de forma (n_samples, n_features_projected).
        Returns:
            torch.Tensor: Escalar con el valor de la pérdida.
        """
        # Distancias por pares en el espacio original y proyectado
        dist_X = pairwise_distances(X)
        dist_Z = pairwise_distances(Z)

        # Pérdida como la diferencia cuadrada entre las distancias
        loss = torch.mean((dist_X - dist_Z)**2)
        return loss