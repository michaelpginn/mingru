"""Torch MinGRU implementation

Christoph Heind, 2024

Based on:
    Were RNNs All We Needed?
    Leo Feng, 2024, https://arxiv.org/pdf/2410.01201v1
"""

import torch
import torch.nn.functional as F


def g(x: torch.Tensor):
    """Proposed activation function for h"""
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))


def log_g(x: torch.Tensor):
    """Proposed activation function for h in log-space"""
    return torch.where(x >= 0, (x + 0.5).log(), -F.softplus(-x))


def parallel_scan_log(log_a, log_b):
    """Parallel scan in log-space.

    Efficiently computes
        x_t = a_t*x_{t-1} + b_t

    Params:
        log_a: (B,T,N) log-coefficients for timestep 1..T
        log_b: (B,T+1,N) log-values of b including x_0

    Returns:
        x: (B,T+1,N) sequence values computed in parallel.

    Based on:
        Efficient Parallelization of a Ubiquitous Sequential Computation
        Franz A. Heinsen, 2023, https://arxiv.org/pdf/2311.06281
    """
    a_star = F.pad(torch.cumsum(log_a, dim=1), (0, 0, 1, 0))
    x0_plus_b_star = torch.logcumsumexp(log_b - a_star, dim=1)
    log_x = a_star + x0_plus_b_star
    return torch.exp(log_x)


class MinGRU(torch.nn.Module):
    """Minimum GRU implementation proposed in 'Were RNNs All We Needed?'.

    Based on the input shapes, automatically dispatches to sequential
    and efficient log-space parallel implementations.
    """

    def __init__(self, input_dims: int, hidden_dims: int):
        super().__init__()
        self.linear_z = torch.nn.Linear(input_dims, hidden_dims)
        self.linear_h = torch.nn.Linear(input_dims, hidden_dims)

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        """Forward function

        When passed a three dimensional input/hidden value
        array, the log-space parallel evaluation is invoked,
        otherwise the sequential mode is called.

        Both functions report equal values.

        *Note* h is supposed to be already activated g(h0). This
        is required if  `h0` contains negative values, as negative
        values are not supported by this implementation. This is only
        required for the initial hidden state, not any subsequent state.

        Params:
            x: (B,input_dims) or (B,S,input_dims) input values
            h: (B,hidden_dims) or (B,1,hidden_dims) initial/previous hidden values

        Returns:
            hnext: (B,hidden_dims) or (B,S,hidden_dims) next hidden values.
        """
        nd_x = x.ndim
        nd_h = h.ndim

        if nd_x == 2 and nd_h == 2:
            return self.forward_sequential(x, h)
        elif nd_x == 3 and nd_h == 3:
            return self.forward_parallel(x, h)
        else:
            raise ValueError(
                "Input shapes should be either both 2 or both 3 dimensional"
            )

    def forward_sequential(self, x: torch.Tensor, h_prev: torch.Tensor):
        """Sequential forward.

        Params:
            x: (B,input_dims) input
            h_prev: (B,hidden_dims) previous hidden dims

        Returns:
            h: (B, hidden_dims) next hidden dims
        """
        z = torch.sigmoid(self.linear_z(x))
        h_tilde = g(self.linear_h(x))
        h_t = (1 - z) * h_prev + z * h_tilde
        return h_t

    def forward_parallel(self, x: torch.Tensor, h_0: torch.Tensor):
        """Parallel forward

        Params:
            x: (B,S,input_dims) input
            h_0: (B,1,hidden_dims) initial hidden-state

        Returns:
            h: (B,S,hidden_dims) hidden states
        """
        k = self.linear_z(x)
        log_z = -F.softplus(-k)  # log(z)
        log_coeffs = -F.softplus(k)  # log(1-z)
        log_h_0 = h_0.log()
        log_tilde_h = log_g(self.linear_h(x))
        h = parallel_scan_log(
            log_coeffs, torch.cat((log_h_0, log_z + log_tilde_h), dim=1)
        )
        return h[:, 1:]  # tail
