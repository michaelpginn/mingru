"""Torch MinGRU implementation

Christoph Heind, 2024

Based on:
    Were RNNs All We Needed?
    Leo Feng, 2024, https://arxiv.org/pdf/2410.01201v1
"""

from typing import Dict
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


def to_gate_hidden_conv2d(
    x: torch.Tensor,
    kernel: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int = 1,
    padding: str = "same",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute gate and hidden outputs using 2D convolutional transform.

    Params:
        x: (B,S,input_dims,H,W) input tensor
        kernel: (hidden_dims*2, input_dims, K, K) kernel
        bias: (hidden_dims*2,) optional bias

    Returns:
        gate: (B,S,hidden_dims,H,W) gate outputs
        hidden: (B,S,hidden_dims,H,W) hidden outputs
    """
    B, S, input_dims, H, W = x.shape
    out_dims = kernel.shape[0]

    gate, hidden = (
        F.conv2d(
            x.view(B * S, input_dims, H, W),
            kernel,
            bias,
            stride=stride,
            padding=padding,
        )
        .view(B, S, out_dims, H, W)
        .chunk(2, dim=2)
    )
    return gate, hidden


def to_gate_hidden_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute gate, hidden outputs using linear transform"""
    gate, hidden = F.linear(x, weight, bias).chunk(2, dim=2)
    return gate, hidden


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


def mingru(
    x: torch.Tensor,
    h: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: str = "same",
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
    if x.ndim == 3:
        gate, hidden = to_gate_hidden_linear(x, weight, bias)
    elif x.ndim == 5:
        gate, hidden = to_gate_hidden_conv2d(
            x, weight, bias, stride=stride, padding=padding
        )
    else:
        raise ValueError(f"Expected input dims to be either 3/5, found {x.ndim}.")

    if x.shape[1] == 1:
        return _mingru_sequential(h, gate, hidden)
    else:
        return _mingru_parallel(h, gate, hidden)


__all__ = ["mingru", "g", "log_g"]
