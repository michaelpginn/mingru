"""PyTorch (convolutional) MinGRU reference implementation

Christoph Heind, 2024
https://github.com/cheind/mingru
"""

from typing import Final

import torch

from . import functional as mF


# Used for dynamic indexing of torch.nn.ModuleList during scripting.
@torch.jit.interface
class ModuleInterface(torch.nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pass


class MinGRUCell(torch.nn.Module):
    """A minimal gated recurrent unit cell."""

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
        """Initialize MinGRU cell.

        Params:
            input_size: number of expected features in input
            hidden_size: number of features in hidden state
            bias: If false, no bias weights will be allocated
            device: optional device for linear layer
            dtype: optional dtype for linear layer
        """
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype, "bias": bias}

        self.to_gate_hidden = torch.nn.Linear(
            input_size,
            hidden_size * 2,  # Compute gate/hidden outputs in tandem
            **factory_kwargs,
        )
        self.layer_sizes = tuple([input_size, hidden_size])

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor | None = None,
    ):
        """Evaluate the MinGRU

        Params:
            x: (B,S,input_size) input features
            h: (B,1,hidden_size) optional previous hidden state
                features

        Returns:
            h': (B,1,hidden_size) next hidden state
        """
        assert (
            x.ndim == 3 and x.shape[2] == self.layer_sizes[0]
        ), "x should be (B,S,input_size)"

        if h is None:
            h = self.init_hidden_state(x)

        gate, hidden = self.to_gate_hidden(x).chunk(2, dim=2)

        hnext = mF.mingru_gate_hidden(gate, hidden, h)
        return hnext

    def init_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        """Returns a 'zero' hidden state."""
        return mF.g(x.new_zeros(x.shape[0], 1, self.layer_sizes[-1]))


class MinGRU(torch.nn.Module):
    """A multi-layer minimal gated recurrent unit (MinGRU)."""

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
        """Initialize MinGRU cell.

        Params:
            input_size: number of expected features in input
            hidden_sizes: list of number of features in each stacked hidden
                state
            bias: If false, no bias weights will be allocated in linear layers
            dropout: If > 0, dropout will be applied to each layer input,
                except for last layer.
            residual: If true, residual connections will be added between each
                layer. If the input/output sizes are different, linear
                adjustment layers will be added.
            device: optional device for linear layer
            dtype: optional dtype for linear layer
        """

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

    def _create_residual_align_layers(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ):
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
        seq_lengths: torch.Tensor | None = None,
    ):
        """Evaluate the MinGRU.

        Params:
            x: (B,S,input_size) input features
            h: optional list of tensors with shape (B,1,hidden_sizes[i])
                containing previous hidden states.
            seq_lengths: (B,) optional tensor with (non-pad token) lengths
                of each sequence in the batch

        Returns:
            out: (B,S,hidden_sizes[-1]) outputs of the last layer
            h': list of tensors with shape (B,1,hidden_sizes[i]) containing
                next hidden states per layer.
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
            if seq_lengths is not None:
                last_hidden_for_layer = out[torch.arange(len(out)), seq_lengths].unsqueeze(1)
            else:
                last_hidden_for_layer = out[:, -1]
            next_hidden.append(last_hidden_for_layer)

            # Add skip connection
            if self.residual:
                # ModuleInterace is required to support dynamic indexing of
                # ModuleLists.
                # See https://github.com/pytorch/pytorch/issues/47496
                al: ModuleInterface = self.residual_layers[lidx]
                out = out + al.forward(inp)

            # Apply dropout (except for last)
            is_not_last = lidx < (self.num_layers - 1)
            if is_not_last and (self.dropout > 0):
                out = torch.nn.functional.dropout(
                    out, p=self.dropout, training=self.training
                )

            # Next input is previous output
            inp = out

        return out, next_hidden

    def init_hidden_state(self, x):
        """Returns a list of 'zero' hidden states for each layer."""
        return [
            mF.g(x.new_zeros(x.shape[0], 1, hidden_size))
            for hidden_size in self.layer_sizes[1:]
        ]


class MinConv2dGRUCell(torch.nn.Module):
    """A minimal convolutional gated recurrent unit cell."""

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
        """Initialize convolutional MinGRU cell.

        Params:
            input_size: number of expected features in input
            hidden_size: number of features in hidden state
            kernel_size: kernel size of convolutional layer
            stride: stride in convolutional layer
            padding: padding in convolutional layer
            bias: If false, no bias weights will be allocated
            device: optional device for linear layer
            dtype: optional dtype for linear layer
        """

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
        """Evaluate the convolutional MinGRU

        Params:
            x: (B,S,input_size,H,W) input features
            h: (B,1,hidden_size,H',W') optional previous
                hidden state features

        Returns:
            h': (B,1,hidden_size,H',W') next hidden state
        """

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
                self.to_gate_hidden(
                    x[:1, :1].flatten(0, 1),
                )
                .unflatten(0, (1, 1))
                .shape[3:]
            )
        return mF.g(x.new_zeros(x.shape[0], 1, self.layer_sizes[-1], H, W))


class MinConv2dGRU(torch.nn.Module):
    """A multi-layer minimal convolutional gated recurrent unit (MinGRU)."""

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
        """Initialize convolutional MinGRU cell.

        Params:
            input_size: number of expected features in input
            hidden_sizes: list containing the number of
                output features per hidden layer.
            kernel_sizes: kernel sizes per convolutional layer
            strides: strides per convolutional layer
            paddings: paddings per convolutional layer
            residual: If true, skip connections between each layer
                are added. If spatials or feature dimensions mismatch,
                necessary alignment convolutions are added.
            bias: If false, no bias weights will be allocated
            device: optional device for linear layer
            dtype: optional dtype for linear layer
        """
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
            # Need to deal with different input/output channels and spatial
            # dims
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
            x: (B,S,input_size,H,W) input of first layer
            h: optional previous/initial hidden states per layer.
                If not given a 'zero' initial state is allocated.
                Shape per layer $i$ is given by (B,1,hidden_size[i],H',W'),
                where W' and H' are determined by the convolution settings.

        Returns:
            out: (B,S,hidden_dims,H',W') outputs of the last layer
            h': list of next hidden states per layer. Shape for layer $i$
                is given by (B,1,hidden_size[i],H',W'), where spatial
                dimensions are determined by convolutional settings.
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
                # ModuleLists.
                # See https://github.com/pytorch/pytorch/issues/47496
                al: ModuleInterface = self.residual_layers[lidx]
                out = out + al.forward(inp.flatten(0, 1)).unflatten(0, (B, S))

            # Apply dropout (except for last)
            is_not_last = lidx < (self.num_layers - 1)
            if is_not_last and (self.dropout > 0):
                out = torch.nn.functional.dropout(
                    out, p=self.dropout, training=self.training
                )

            # Next input is previous output
            inp = out

        return out, next_hidden

    def init_hidden_state(self, x: torch.Tensor) -> list[torch.Tensor]:
        hs = []
        B = x.shape[0]
        # We dynamically determine the required hidden shapes, to avoid
        # fiddling with spatial dimension computation. This just uses
        # the first sequence element from the first batch todo so, and hence
        # should not lead to major performance impact.
        with torch.no_grad():
            # Cannot make the following a reusable function because
            # nn.Modules are not accepted as parameters in scripting...
            for lidx, gh in enumerate(self.to_gate_hidden):
                y, _ = (
                    gh(x[:1, :1].flatten(0, 1))
                    .unflatten(
                        0,
                        (1, 1),
                    )
                    .chunk(2, dim=2)
                )
                h = mF.g(y.new_zeros(B, 1, y.shape[2], y.shape[3], y.shape[4]))
                hs.append(h)
                x = y
        return hs


__all__ = ["MinGRUCell", "MinGRU", "MinConv2dGRUCell", "MinConv2dGRU"]
