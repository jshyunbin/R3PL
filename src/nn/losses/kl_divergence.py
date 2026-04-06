import torch
from typing import Tuple

def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculates KL Divergence of variational distribution to N(0, 1) prior distribution

    Args:
        mu (torch.Tensor): mean value tensor of shape (batch_size, latent_dim)
        logvar (torch.Tensor): log variance tensor of shape (batch_size, latent_dim)

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: total loss, dimension wise loss and mean loss respectively
    """
    batch_size = mu.shape[0]
    assert batch_size != 0

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld