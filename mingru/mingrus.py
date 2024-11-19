"""Torch MinGRU implementation

Christoph Heind, 2024

Based on:
    Were RNNs All We Needed?
    Leo Feng, 2024, https://arxiv.org/pdf/2410.01201v1
"""

from typing import Final
import torch

from . import functional as mF


class MinGRUBase(torch.nn.Module):
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
        dropout: float,
        residual: bool,
        gate_hidden_layers: tuple[torch.nn.Module, ...],
        residual_input_align: torch.nn.Module | None,
    ):
        """Initialize MinGRU"""

        super().__init__()

        self.gate_hidden_layers = torch.nn.ModuleList(gate_hidden_layers)
        self.residual_input_align = residual_input_align
        self.num_layers = len(self.gate_hidden_layers)
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.residual = residual

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

        assert (x.ndim == 3 or x.ndim == 5) and x.shape[
            2
        ] == self.input_dims, "x should be (B,S,input_dims or (B,S,input_dims,H,W)"

        if h is None:
            h = self.init_zero_hidden_state(x)
        else:
            h = self.expand_hidden_state(x, h)
            # Note, we don't apply g() in this case, we assume it has been
            # applied, otherwise we have inconsistencies between sequential
            # and parallel mode.

        # input to next layer
        inp = x
        final_hidden_per_layer = []

        # hidden states across layers
        for lidx, gh_layer in enumerate(self.gate_hidden_layers):
            h_prev = h[lidx]

            # (B,S,hidden_dims) or (B,S,hidden_dims,H,W)
            out = mF.mingru(inp, h_prev, gh_layer.weight, gh_layer.bias)

            # Save final hidden state of layer
            final_hidden_per_layer.append(out[:, -1:])

            # Add skip connection
            if self.residual:
                f = self.residual_input_align(inp) if lidx == 0 else inp
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

    def init_zero_hidden_state(self, x: torch.Tensor):
        raise NotImplementedError()

    def expand_hidden_state(self, x: torch.Tensor, h: torch.Tensor):
        raise NotImplementedError()


class MinGRU(MinGRUBase):
    def __init__(
        self,
        input_dims,
        hidden_dims,
        *,
        num_layers: int = 1,
        dropout: float = 0.0,
        residual: bool = False,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):

        factory_kwargs = {"device": device, "dtype": dtype, "bias": bias}

        dims = [input_dims] + [hidden_dims] * num_layers
        gate_hidden_layers = []
        for ind, outd in zip(dims[:-1], dims[1:]):
            # combine linear gate and hidden transform
            gh = torch.nn.Linear(ind, outd * 2, **factory_kwargs)
            gate_hidden_layers.append(gh)

        align_layer = None
        if residual:
            if input_dims != hidden_dims:
                align_layer = torch.nn.Linear(input_dims, hidden_dims, **factory_kwargs)
            else:
                align_layer = torch.nn.Identity()

        super().__init__(
            input_dims,
            hidden_dims,
            dropout,
            residual,
            gate_hidden_layers,
            align_layer,
        )

    def init_zero_hidden_state(self, x):
        h = x.new_zeros((self.num_layers, x.shape[0], 1, self.hidden_dims))
        h = mF.g(h)
        return h

    def expand_hidden_state(self, x, h):
        h = h.expand(self.num_layers, x.shape[0], 1, self.hidden_dims)
        return h


class MinConv2dGRU(MinGRUBase):
    def __init__(
        self,
        input_dims,
        hidden_dims,
        *,
        num_layers: int = 1,
        kernel_size: int = 3,
        dropout: float = 0.0,
        residual: bool = False,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):

        factory_kwargs = {
            "device": device,
            "dtype": dtype,
            "bias": bias,
            "stride": 1,
            "padding": 0,
        }

        dims = [input_dims] + [hidden_dims] * num_layers
        gate_hidden_layers = []
        for ind, outd in zip(dims[:-1], dims[1:]):
            # combine linear gate and hidden transform
            gh = torch.nn.Conv2d(ind, outd * 2, kernel_size, **factory_kwargs)
            gate_hidden_layers.append(gh)

        align_layer = None
        if residual:
            if input_dims != hidden_dims:

                class Align(torch.nn.Module):
                    def __init__(
                        self,
                    ):
                        super().__init__()
                        self.conv = torch.nn.Conv2d(
                            input_dims,
                            hidden_dims,
                            kernel_size=1,
                            **factory_kwargs,
                        )

                    def forward(self, x: torch.Tensor) -> torch.Tensor:
                        B, S, _, H, W = x.shape
                        x = self.conv(x.view(B * S, input_dims, H, W)).view(
                            B, S, hidden_dims, H, W
                        )
                        return x

                align_layer = Align()
            else:
                align_layer = torch.nn.Identity()

        super().__init__(
            input_dims,
            hidden_dims,
            dropout,
            residual,
            gate_hidden_layers,
            align_layer,
        )

    def init_zero_hidden_state(self, x):
        B, H, W = x.shape[0], x.shape[-2], x.shape[-1]
        h = x.new_zeros((self.num_layers, B, 1, self.hidden_dims, H, W))
        h = mF.g(h)
        return h

    def expand_hidden_state(self, x, h):
        B, H, W = x.shape[0], x.shape[-2], x.shape[-1]
        h = h.expand(self.num_layers, B, 1, self.hidden_dims, H, W)
        return h
