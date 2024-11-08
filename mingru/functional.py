"""Torch MinGRU implementation

Christoph Heind, 2024

Based on:
    Were RNNs All We Needed?
    Leo Feng, 2024, https://arxiv.org/pdf/2410.01201v1
"""

import torch
import torch.nn.functional as F

from .scan import parallel_scan_log


def g(x: torch.Tensor):
    """Proposed activation function for h"""
    out = torch.empty_like(x)
    mask = x >= 0
    out[mask] = x[mask] + 0.5
    out[~mask] = torch.sigmoid(x[~mask])
    return out


def log_g(x: torch.Tensor):
    """Proposed activation function for h in log-space"""
    out = torch.empty_like(x)
    mask = x >= 0
    out[mask] = (x[mask] + 0.5).log()
    out[~mask] = -F.softplus(-x[~mask])
    return out


def mingru_sequential(
    x: torch.Tensor,
    h: torch.Tensor,
    weight_z: torch.Tensor,
    weight_h: torch.Tensor,
    bias_z: torch.Tensor | None = None,
    bias_h: torch.Tensor | None = None,
):
    """Sequential forward.

    Params:
        x: (B,1,input_dims) input
        h: (B,1,hidden_dims) previous hidden dims

    Returns:
        h: (B,1,hidden_dims) next hidden dims
    """
    z = torch.sigmoid(F.linear(x, weight_z, bias_z))
    h_tilde = g(F.linear(x, weight_h, bias_h))
    h_t = (1 - z) * h + z * h_tilde
    return h_t


def mingru_parallel(
    x: torch.Tensor,
    h: torch.Tensor,
    weight_z: torch.Tensor,
    weight_h: torch.Tensor,
    bias_z: torch.Tensor | None = None,
    bias_h: torch.Tensor | None = None,
):
    """Parallel forward

    Params:
        x: (B,S,input_dims) input
        h: (B,1,hidden_dims) initial hidden-state

    Returns:
        h: (B,S,hidden_dims) hidden states
    """
    k = F.linear(x, weight_z, bias_z)
    log_z = -F.softplus(-k)  # log(z)
    log_coeffs = -F.softplus(k)  # log(1-z)
    log_h_0 = h.log()
    log_tilde_h = log_g(F.linear(x, weight_h, bias_h))
    h = parallel_scan_log(log_coeffs, torch.cat((log_h_0, log_z + log_tilde_h), dim=1))
    return h[:, 1:]  # tail


def mingru(
    x: torch.Tensor,
    h: torch.Tensor,
    weight_z: torch.Tensor,
    weight_h: torch.Tensor,
    bias_z: torch.Tensor | None = None,
    bias_h: torch.Tensor | None = None,
):
    S = x.shape[1]
    fn = mingru_sequential if S == 1 else mingru_parallel
    return fn(x, h, weight_z, weight_h, bias_z, bias_h)


__all__ = ["mingru", "g", "log_g"]
