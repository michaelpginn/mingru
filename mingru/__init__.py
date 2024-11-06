"""Torch MinGRU implementation

Christoph Heind, 2024

Based on:
    Were RNNs All We Needed?
    Leo Feng, 2024, https://arxiv.org/pdf/2410.01201v1
"""

import math
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

    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        *,
        num_layers: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        """Initialize MinGRU

        Params:
            input_dims: number of input dimensions
            hidden_dims: number of hidden dimensions
            num_layers: number of layers. When > 1, the inputs
                of layer l is the output of layer l-1
            dropout: when > 0, applies dropout to inputs except
                for last layer
            bias: when true, linear transformations have a bias term
            device: optional device
            dtype: optional dtype
        """

        super().__init__()

        assert batch_first, "Batch-first is currently required"

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = True
        self.dropout = dropout

        dims_z = [input_dims] + [hidden_dims] * num_layers
        dims_h = [input_dims] + [hidden_dims] * num_layers

        factory_kwargs = {"device": device, "dtype": dtype, "bias": bias}

        layers = []
        for ind, outd in zip(dims_z[:-1], dims_z[1:]):
            n = torch.nn.Linear(ind, outd, **factory_kwargs)
            layers.append(self._init_linear(n))
        self.linear_z = torch.nn.ModuleList(layers)

        layers = []
        for ind, outd in zip(dims_h[:-1], dims_h[1:]):
            n = torch.nn.Linear(ind, outd, **factory_kwargs)
            layers.append(self._init_linear(n))
        self.linear_h = torch.nn.ModuleList(layers)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor | None = None,
        *,
        return_all_outputs: bool = False,
    ):
        """Evaluate the MinGRU.

        Params:
            x: (B,S,input_dims) input of first layer
            h: (L,B,1,hidden_dims) or expandable shape of initial hidden
                values per layer.

        Returns:
            h: (B,S,hidden_dims) or (L,B,S,hidden_dims) when `return_all_outputs` is true.
        """

        assert (
            x.ndim == 3 and x.shape[-1] == self.input_dims
        ), "x should be (B,S,input_dims)"

        B, S, _ = x.shape
        if h is None:
            h = x.new_zeros((self.num_layers, B, 1, self.hidden_dims))
            h = g(h)
        else:
            h = h.expand(self.num_layers, B, 1, self.hidden_dims)
            # Note, we don't apply h in this case, we assume it has been
            # applied, otherwise we have inconsistencies between sequential
            # and parallel mode.

        fwdfn = self.forward_sequential if S == 1 else self.forward_parallel

        inp = x
        outs = []
        for lidx, (lin_z, lin_h, h0) in enumerate(
            zip(
                self.linear_z,
                self.linear_h,
                h,
            )
        ):
            out = fwdfn(inp, h0, lin_z, lin_h)
            inp = out
            if lidx < (self.num_layers - 1):
                inp = torch.bernoulli(torch.full_like(out, 1 - self.dropout))
            outs.append(out)

        if return_all_outputs:
            return torch.stack(outs, 0)
        else:
            return outs[-1]

    def forward_sequential(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        lin_z: torch.nn.Linear,
        lin_h: torch.nn.Linear,
    ):
        """Sequential forward.

        Params:
            x: (B,1,input_dims) input
            h: (B,1,hidden_dims) previous hidden dims

        Returns:
            h: (B,1,hidden_dims) next hidden dims
        """
        z = torch.sigmoid(lin_z(x))
        h_tilde = g(lin_h(x))
        h_t = (1 - z) * h + z * h_tilde
        return h_t

    def forward_parallel(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        lin_z: torch.nn.Linear,
        lin_h: torch.nn.Linear,
    ):
        """Parallel forward

        Params:
            x: (B,S,input_dims) input
            h: (B,1,hidden_dims) initial hidden-state

        Returns:
            h: (B,S,hidden_dims) hidden states
        """
        k = lin_z(x)
        log_z = -F.softplus(-k)  # log(z)
        log_coeffs = -F.softplus(k)  # log(1-z)
        log_h_0 = h.log()
        log_tilde_h = log_g(lin_h(x))
        h = parallel_scan_log(
            log_coeffs, torch.cat((log_h_0, log_z + log_tilde_h), dim=1)
        )
        return h[:, 1:]  # tail

    def _init_linear(self, n: torch.nn.Linear):
        stdv = 1.0 / math.sqrt(n.weight.size(1))
        n.weight.data.uniform_(-stdv, stdv)
        if n.bias is not None:
            n.bias.data.uniform_(-stdv, stdv)
        return n
