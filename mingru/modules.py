"""PyTorch (convolutional) MinGRU reference implementation

Christoph Heind, 2024
https://github.com/cheind/mingru

Based on:
    Were RNNs All We Needed?
    Leo Feng, 2024, https://arxiv.org/pdf/2410.01201v1
"""

from typing import Final

import torch

from . import functional as mF


@torch.jit.interface
class ModuleInterface(torch.nn.Module):
    def forward(
        self, input: torch.Tensor
    ) -> torch.Tensor:  # `input` has a same name in Sequential forward
        pass


class MinGRUCell(torch.nn.Module):
    layer_sizes: Final[tuple[int, ...]]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype, "bias": bias}

        self.to_gate_hidden = torch.nn.Linear(
            input_size,
            hidden_size * 2,
            **factory_kwargs,
        )
        self.layer_sizes = tuple([input_size, hidden_size])

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor | None = None,
    ):
        assert (
            x.ndim == 3 and x.shape[2] == self.layer_sizes[0]
        ), "x should be (B,S,input_size)"

        if h is None:
            h = self.init_hidden_state(x)

        gate, hidden = self.to_gate_hidden(x).chunk(2, dim=2)

        hnext = mF.mingru_gate_hidden(gate, hidden, h)
        return hnext

    def init_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        return mF.g(x.new_zeros(x.shape[0], 1, self.layer_sizes[-1]))


class MinGRU(torch.nn.Module):

    layer_sizes: Final[tuple[int, ...]]
    num_layers: Final[int]
    residual: Final[bool]
    dropout: Final[float]

    def __init__(
        self,
        input_size: int,
        hidden_sizes: int | list[int],
        *,
        bias: bool = True,
        dropout: float = 0.0,
        residual: bool = False,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()

        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        self.layer_sizes = tuple([input_size] + hidden_sizes)
        self.num_layers = len(hidden_sizes)
        self.dropout = max(min(dropout, 1.0), 0.0)
        self.residual = residual

        self.to_gate_hidden = self._create_linear_gate_hidden_layers(
            device, dtype, bias
        )
        if residual:
            self.residual_layers = self._create_residual_align_layers(
                device,
                dtype,
            )
        else:
            # Needed for scripting
            self.residual_layers = torch.nn.ModuleList()

    def _create_linear_gate_hidden_layers(
        self, device: torch.device, dtype: torch.dtype, bias: bool
    ):
        factory_kwargs = {"device": device, "dtype": dtype, "bias": bias}
        layers = []
        # combine linear gate and hidden transform
        for ind, outd in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            gh = torch.nn.Linear(ind, outd * 2, **factory_kwargs)
            layers.append(gh)
        return torch.nn.ModuleList(layers)

    def _create_residual_align_layers(self, device: torch.device, dtype: torch.dtype):
        factory_kwargs = {"device": device, "dtype": dtype, "bias": False}
        layers = []
        for ind, outd in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            if ind != outd:
                al = torch.nn.Linear(ind, outd, **factory_kwargs)
            else:
                al = torch.nn.Identity()
            layers.append(al)
        return torch.nn.ModuleList(layers)

    def forward(
        self,
        x: torch.Tensor,
        h: list[torch.Tensor] | None = None,
    ):
        """Evaluate the MinGRU.

        Params:
            x: (B,S,input_size) input of first layer
            h: optional list[(B,1,hidden_size)] previous/initial
                hidden state of each layer. If not given a 'zero'
                initial state is allocated.

        Returns:
            out: (B,S,hidden_dims) outputs of the last layer
            h: (num_layers,B,1,hidden_dims) containing the final hidden state
                for the input sequence.
        """
        assert (
            x.ndim == 3 and x.shape[2] == self.layer_sizes[0]
        ), "x should be (B,S,input_size)"

        if h is None:
            h = self.init_hidden_state(x)

        # input to next layer
        inp = x
        next_hidden = []

        # hidden states across layers
        for lidx, gh_layer in enumerate(self.to_gate_hidden):
            h_prev = h[lidx]

            gate, hidden = gh_layer(inp).chunk(2, dim=2)
            out = mF.mingru_gate_hidden(gate, hidden, h_prev)
            # (B,S,hidden_dims)

            # Save final hidden state of layer
            next_hidden.append(out[:, -1:])

            # Add skip connection
            if self.residual:
                # ModuleInterace is required to support dynamic indexing of
                # ModuleLists. See https://github.com/pytorch/pytorch/issues/47496
                al: ModuleInterface = self.residual_layers[lidx]
                out = out + al.forward(inp)

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

        return out, next_hidden

    def init_hidden_state(self, x):
        return [
            mF.g(x.new_zeros(x.shape[0], 1, hidden_size))
            for hidden_size in self.layer_sizes[1:]
        ]


class MinConv2dGRUCell(torch.nn.Module):
    layer_sizes: Final[tuple[int, ...]]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()

        factory_kwargs = {
            "device": device,
            "dtype": dtype,
            "bias": bias,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
        }

        self.to_gate_hidden = torch.nn.Conv2d(
            input_size,
            hidden_size * 2,
            **factory_kwargs,
        )
        self.layer_sizes = tuple([input_size, hidden_size])

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor | None = None,
    ):
        assert (
            x.ndim == 5 and x.shape[2] == self.layer_sizes[0]
        ), "x should be (B,S,input_size,H,W)"

        if h is None:
            h = self.init_hidden_state(x)

        B, S = x.shape[:2]
        gate, hidden = (
            self.to_gate_hidden(x.flatten(0, 1))
            .unflatten(
                0,
                (B, S),
            )
            .chunk(2, dim=2)
        )

        hnext = mF.mingru_gate_hidden(gate, hidden, h)
        return hnext

    def init_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        B, S = x.shape[:2]
        with torch.no_grad():
            H, W = (
                self.to_gate_hidden(x.flatten(0, 1))
                .unflatten(
                    0,
                    (B, S),
                )
                .shape[3:]
            )
        return mF.g(x.new_zeros(x.shape[0], 1, self.layer_sizes[-1], H, W))


class AlignConvBlock(torch.nn.Module):
    def __init__(self, in_size: int, out_size: int, kernel: int, stride: int):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_size,
            out_size,
            kernel_size=kernel,
            stride=stride,
            bias=False,
            padding="same",
        )
        self.in_size = in_size
        self.out_size = out_size

    def forward(self, x: torch.Tensor):
        B, S, _, H, W = x.shape
        x = self.conv(x.view(B * S, self.in_size, H, W)).view(B, S, self.out_size, H, W)
        return x


class MinConv2dGRU(torch.nn.Module):

    layer_sizes: Final[tuple[int, ...]]
    num_layers: Final[int]
    residual: Final[bool]
    dropout: Final[float]

    def __init__(
        self,
        input_size: int,
        hidden_sizes: int | list[int],
        kernel_sizes: int | list[int],
        *,
        strides: int | list[int] = 1,
        paddings: int | list[int] = 0,
        dropout: float = 0.0,
        residual: bool = False,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()

        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(hidden_sizes)

        if isinstance(strides, int):
            strides = [strides] * len(hidden_sizes)

        if isinstance(paddings, int):
            paddings = [paddings] * len(hidden_sizes)

        self.layer_sizes = tuple([input_size] + hidden_sizes)
        self.num_layers = len(hidden_sizes)
        self.dropout = max(min(dropout, 1.0), 0.0)
        self.residual = residual

        self.to_gate_hidden = self._create_conv_gate_hidden_layers(
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            device=device,
            dtype=dtype,
            bias=bias,
        )
        if residual:
            self.residual_layers = self._create_residual_align_layers(
                kernel_sizes=kernel_sizes,
                strides=strides,
                paddings=paddings,
                device=device,
                dtype=dtype,
            )
        else:
            # Needed for scripting
            self.residual_layers = torch.nn.ModuleList()

    def _create_conv_gate_hidden_layers(
        self,
        kernel_sizes: list[int],
        strides: list[int],
        paddings: list[int],
        device: torch.device,
        dtype: torch.dtype,
        bias: bool,
    ):
        factory_kwargs = {"device": device, "dtype": dtype, "bias": bias}
        layers = []
        for lidx, (ind, outd) in enumerate(
            zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        ):
            gh = torch.nn.Conv2d(
                ind,
                outd * 2,
                kernel_size=kernel_sizes[lidx],
                stride=strides[lidx],
                padding=paddings[lidx],
                **factory_kwargs,
            )
            layers.append(gh)
        return torch.nn.ModuleList(layers)

    def _create_residual_align_layers(
        self,
        kernel_sizes: list[int],
        strides: list[int],
        paddings: list[int],
        device: torch.device,
        dtype: torch.dtype,
    ):
        factory_kwargs = {"device": device, "dtype": dtype, "bias": False}
        layers = []
        for lidx, (ind, outd) in enumerate(
            zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        ):
            # Need to deal with different input/output channels and spatial dims
            with torch.no_grad():
                x = torch.randn(1, ind, 16, 16)
                y = self.to_gate_hidden[lidx](x)
            if ind != outd or x.shape[2:] != y.shape[2:]:
                al = torch.nn.Conv2d(
                    ind,
                    outd,
                    kernel_size=kernel_sizes[lidx],
                    stride=strides[lidx],
                    padding=paddings[lidx],
                    **factory_kwargs,
                )
            else:
                al = torch.nn.Identity()
            layers.append(al)
        return torch.nn.ModuleList(layers)

    def forward(
        self,
        x: torch.Tensor,
        h: list[torch.Tensor] | None = None,
    ):
        """Evaluate the MinGRU.

        Params:
            x: (B,S,input_size) input of first layer
            h: optional list[(B,1,hidden_size)] previous/initial
                hidden state of each layer. If not given a 'zero'
                initial state is allocated.

        Returns:
            out: (B,S,hidden_dims) outputs of the last layer
            h: (num_layers,B,1,hidden_dims) containing the final hidden state
                for the input sequence.
        """
        assert (
            x.ndim == 5 and x.shape[2] == self.layer_sizes[0]
        ), "x should be (B,S,input_size)"

        if h is None:
            h = self.init_hidden_state(x)

        # input to next layer
        inp = x
        next_hidden = []

        # hidden states across layers
        for lidx, gh_layer in enumerate(self.to_gate_hidden):
            h_prev = h[lidx]

            B, S = inp.shape[:2]
            gate, hidden = (
                gh_layer(inp.flatten(0, 1))
                .unflatten(
                    0,
                    (B, S),
                )
                .chunk(2, dim=2)
            )
            out = mF.mingru_gate_hidden(gate, hidden, h_prev)
            # (B,S,hidden_dims)

            # Save final hidden state of layer
            next_hidden.append(out[:, -1:])

            # Add skip connection
            if self.residual:
                # ModuleInterace is required to support dynamic indexing of
                # ModuleLists. See https://github.com/pytorch/pytorch/issues/47496
                al: ModuleInterface = self.residual_layers[lidx]
                out = out + al.forward(inp.flatten(0, 1)).unflatten(0, (B, S))

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

        return out, next_hidden

    def init_hidden_state(self, x: torch.Tensor) -> list[torch.Tensor]:
        hs = []
        B = x.shape[0]
        with torch.no_grad():
            # Cannot make the following a reusable function because
            # nn.Modules are not accepted as parameters in scripting...
            for lidx, gh in enumerate(self.to_gate_hidden):
                y, _ = gh(x[:1, :1].flatten(0, 1)).unflatten(0, (1, 1)).chunk(2, dim=2)
                h = mF.g(y.new_zeros(B, 1, y.shape[2], y.shape[3], y.shape[4]))
                hs.append(h)
                x = y
        return hs


__all__ = ["MinGRUCell", "MinGRU", "MinConv2dGRUCell", "MinConv2dGRU"]
