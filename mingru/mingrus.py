"""Torch MinGRU implementation

Christoph Heind, 2024

Based on:
    Were RNNs All We Needed?
    Leo Feng, 2024, https://arxiv.org/pdf/2410.01201v1
"""

import math
import torch
import torch.nn.functional as F

from . import functional as mF


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
            h: (B,S,hidden_dims) or (L,B,S,hidden_dims) when
                `return_all_outputs` is true.
        """

        assert (
            x.ndim == 3 and x.shape[-1] == self.input_dims
        ), "x should be (B,S,input_dims)"

        B = x.shape[0]
        if h is None:
            h = x.new_zeros((self.num_layers, B, 1, self.hidden_dims))
            h = mF.g(h)
        else:
            h = h.expand(self.num_layers, B, 1, self.hidden_dims)
            # Note, we don't apply g() in this case, we assume it has been
            # applied, otherwise we have inconsistencies between sequential
            # and parallel mode.

        inp = x
        outs = []
        for lidx, (lin_z, lin_h, h0) in enumerate(
            zip(
                self.linear_z,
                self.linear_h,
                h,
            )
        ):
            out = mF.mingru(
                inp,
                h0,
                lin_z.weight,
                lin_h.weight,
                lin_z.bias,
                lin_h.bias,
            )
            inp = out
            if (lidx < (self.num_layers - 1)) and (self.dropout > 0):
                inp = inp * torch.bernoulli(
                    torch.full_like(
                        out,
                        1 - self.dropout,
                    )
                )
            outs.append(out)

        if return_all_outputs:
            return torch.stack(outs, 0)
        else:
            return outs[-1]

    def create_chunked_helper(self, h0: torch.Tensor = None):
        """Returns a helper function for sequential evaluation of the RNN.

        Params:
            h0: (B,1,hidden_dims) or (L,B,1,hidden_dims) optional initial
                hidden state

        Returns:
            fn: A stateful closure that takes $x_t$ and calls the rnn with
                $x_t,h_{t-1}$, reports $h_t$ and then updates
                its internal state to $h_t$.
        """
        state = [h0]

        def forward(
            x: torch.Tensor,
            h: torch.Tensor = None,
            return_all_outputs: bool = False,
        ):
            hin = state[-1] if h is None else h
            h = self.forward(x, hin, return_all_outputs=True)
            state[0] = h
            return h if return_all_outputs else h[-1]

        return forward

    def _init_linear(self, n: torch.nn.Linear):
        stdv = 1.0 / math.sqrt(n.weight.size(1))
        n.weight.data.uniform_(-stdv, stdv)
        if n.bias is not None:
            n.bias.data.uniform_(-stdv, stdv)
        return n
