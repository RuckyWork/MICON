"""Training objective: L = lambda_DP * L_DP + lambda_VAR * L_VAR + lambda_COV * L_COV."""

import torch
import torch.nn.functional as F


def variance_loss(x, gamma=1.0, eps=1e-4):
    """Hinge on the standard deviation of each latent dimension, x: (n, C)."""
    std = torch.sqrt(x.var(dim=0) + eps)
    return F.relu(gamma - std).mean()


def covariance_loss(x):
    """Sum of squared off-diagonal covariances divided by C, x: (n, C)."""
    n, c = x.shape
    x = x - x.mean(dim=0)
    cov = (x.T @ x) / (n - 1)
    off_diagonal = cov * (1 - torch.eye(c, device=x.device))
    return off_diagonal.pow(2).sum() / c


def micon_loss(R_c, R_s, predictor, lambda_dp=1.0, lambda_var=1.0, lambda_cov=1.0):
    """R_c, R_s: (B, N, C). Every (slot, region) instance in the batch is one sample."""
    L_dp = predictor(R_c, R_s).mean()
    flat_c = R_c.reshape(-1, R_c.shape[-1])
    flat_s = R_s.reshape(-1, R_s.shape[-1])
    L_var = (variance_loss(flat_c) + variance_loss(flat_s)) / 2
    L_cov = (covariance_loss(flat_c) + covariance_loss(flat_s)) / 2
    loss = lambda_dp * L_dp + lambda_var * L_var + lambda_cov * L_cov
    return loss, {"loss": loss.item(), "L_dp": L_dp.item(), "L_var": L_var.item(), "L_cov": L_cov.item()}
