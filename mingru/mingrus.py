"""Torch MinGRU implementation

Christoph Heind, 2024

Based on:
    Were RNNs All We Needed?
    Leo Feng, 2024, https://arxiv.org/pdf/2410.01201v1
"""

from typing import Final
import math
import torch

from . import functional as mF


class MinGRU(torch.nn.Module):
    """Minimum GRU implementation proposed in 'Were RNNs All We Needed?'"""

    num_layers: Final[int]
    input_dims: Final[int]
    hidden_dims: Final[int]
    dropout: Final[float]
    residual: Final[bool]

    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        *,
        num_layers: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        residual: bool = False,
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
            residual: when true, adds residual connections between layers
            device: optional device
            dtype: optional dtype
        """

        super().__init__()

        assert batch_first, "Batch-first is currently required"

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.residual = residual

        dims = [input_dims] + [hidden_dims] * num_layers

        factory_kwargs = {"device": device, "dtype": dtype, "bias": bias}

        layers = []
        for ind, outd in zip(dims[:-1], dims[1:]):
            # combine linear gate and hidden transform
            n = torch.nn.Linear(ind, outd * 2, **factory_kwargs)
            layers.append(self._init_linear(n))
        self.linear_gate_hidden = torch.nn.ModuleList(layers)

        self.input_residual_align = None
        if self.residual:
            self.input_residual_align = torch.nn.Linear(
                input_dims, hidden_dims, **factory_kwargs
            )

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor | None = None,
    ):
        """Evaluate the MinGRU.

        Params:
            x: (B,S,input_dims) input of first layer
            h: (num_layers,B,1,hidden_dims) or any expandable shape of initial
                /previous hidden values per layer.

        Returns:
            out: (B,S,hidden_dims) outputs of the last layer
            h: (num_layers,B,1,hidden_dims) containing the final hidden state
                for the input sequence.
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

        # input to next layer
        inp = x
        final_hidden_per_layer = []

        # hidden states across layers
        for lidx, linear in enumerate(self.linear_gate_hidden):
            h_prev = h[lidx]

            # (B,S,hidden_dims)
            out = mF.mingru(inp, h_prev, linear.weight, linear.bias)

            # Save final hidden state of layer
            final_hidden_per_layer.append(out[:, -1:])

            # Add skip connection
            if self.residual:
                f = self.input_residual_align(inp) if lidx == 0 else inp
                out = out + f

            # Apply dropout (except for last)
            is_not_last = lidx < (self.num_layers - 1)
            if is_not_last and (self.dropout > 0):
                out = out * torch.bernoulli(
                    torch.full_like(
                        out,
                        1 - self.dropout,
                    )
                )

            # Next input is previous output
            inp = out

        return out, torch.stack(final_hidden_per_layer)

    def _init_linear(self, n: torch.nn.Linear):
        stdv = 1.0 / math.sqrt(n.weight.size(1))
        n.weight.data.uniform_(-stdv, stdv)
        if n.bias is not None:
            n.bias.data.uniform_(-stdv, stdv)
        return n
