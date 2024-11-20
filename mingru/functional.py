"""PyTorch (convolutional) MinGRU reference implementation

Christoph Heind, 2024
https://github.com/cheind/mingru

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


def _mingru_parallel(
    h: torch.Tensor,
    gate: torch.Tensor,
    hidden: torch.Tensor,
):
    """Parallel forward

    Params:
        x: (B,S,input_dims,H,W) input
        h: (B,1,hidden_dims,H,W) initial hidden-state
        kernel: (hidden_dims*2, input_dims, k1, k2)
        bias: (hidden_dims*2,)

    Returns:
        h: (B,S,hidden_dims,H,W) hidden states
    """

    log_z = -F.softplus(-gate)  # log(z)
    log_coeffs = -F.softplus(gate)  # log(1-z)
    log_h_0 = h.log()
    log_tilde_h = log_g(hidden)

    h = parallel_scan_log(
        log_coeffs,
        torch.cat((log_h_0, log_z + log_tilde_h), dim=1),
    )
    return h[:, 1:]  # tail


def _mingru_sequential(
    h: torch.Tensor,
    gate: torch.Tensor,
    hidden: torch.Tensor,
):
    """Sequential forward.

    Params:
        x: (B,1,input_dims,H,W) input
        h: (B,1,hidden_dims,H,W) previous hidden dims
        weight: (hidden_dims*2, input_dims, k1, k2)
        bias: (hidden_dims*2,)

    Returns:
        h: (B,1,hidden_dims,H,W) next hidden dims
    """

    z = torch.sigmoid(gate)
    h_tilde = g(hidden)
    h_t = (1 - z) * h + z * h_tilde
    return h_t


def mingru_gate_hidden(
    gate: torch.Tensor,
    hidden: torch.Tensor,
    h: torch.Tensor,
):
    """Evaluate the (convolutional) MinGRU

    This method is the main entry point to evaluate the MinGRU. It
    combines support for linear and convolutional MinGRUs and is
    scriptable.

    The code dispatches to linear/convolutional transforms based
    on the number of input dimensions 3/5. The code also dispatches
    between sequential and parallel model depending on the size of
    the sequence dimension.

    Params:
        x: (B,S,input_dims) or (B,S,input_dims,H,W) input
        h: (B,1,hidden_dims) or (B,1,hidden_dims,H,W) initial/previous
            hidden state
        weight: (hidden_dims*2, input_dims) or (hidden_dims*2, input_dims, K, K)
            weights of linear/convolution z-gate and hidden transform combined
        bias: (hidden_dims*2,) optional bias term of z-gate
            and hidden transform combined

    Returns:
        h: (B,S,hidden_dims) or (B,S,hidden_dims,H,W) next hidden states
    """

    if gate.shape[1] == 1:
        return _mingru_sequential(h, gate, hidden)
    else:
        return _mingru_parallel(h, gate, hidden)


__all__ = ["mingru_gate_hidden", "g", "log_g"]
