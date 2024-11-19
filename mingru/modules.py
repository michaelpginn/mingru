import torch
from typing import Final

from . import functional as mF


@torch.jit.interface
class ModuleInterface(torch.nn.Module):
    def forward(
        self, input: torch.Tensor
    ) -> torch.Tensor:  # `input` has a same name in Sequential forward
        pass


class MinGRU(torch.nn.Module):

    layer_sizes: Final[tuple[int, ...]]
    num_layers: Final[int]

    def __init__(
        self,
        input_size: int,
        hidden_sizes: int | list[int],
        *,
        dropout: float = 0.0,
        residual: bool = False,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()

        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        factory_kwargs = {"device": device, "dtype": dtype, "bias": bias}

        self.layer_sizes = [input_size] + hidden_sizes
        self.num_layers = len(hidden_sizes)
        gh_layers = []
        align_layers = []
        for ind, outd in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            # combine linear gate and hidden transform
            gh = torch.nn.Linear(ind, outd * 2, **factory_kwargs)
            gh_layers.append(gh)
            # generate align layers for residual connections
            if ind != outd:
                al = torch.nn.Linear(ind, outd, **factory_kwargs)
            else:
                al = torch.nn.Identity()
            align_layers.append(al)
        self.gate_hidden_layers = torch.nn.ModuleList(gh_layers)
        if residual:
            self.align_layers = torch.nn.ModuleList(align_layers)

        self.dropout = max(min(dropout, 1.0), 0.0)
        self.residual = residual

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
        for lidx, gh_layer in enumerate(self.gate_hidden_layers):
            h_prev = h[lidx]

            out = mF.mingru(inp, h_prev, gh_layer.weight, gh_layer.bias)
            # (B,S,hidden_dims)

            # Save final hidden state of layer
            next_hidden.append(out[:, -1:])

            # Add skip connection
            if self.residual:
                # ModuleInterace is required to support dynamic indexing of
                # ModuleLists. See https://github.com/pytorch/pytorch/issues/47496
                al: ModuleInterface = self.align_layers[lidx]
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

    def __init__(
        self,
        input_size: int,
        hidden_sizes: int | list[int],
        *,
        kernel_sizes: int | list[int] = 3,
        strides: int | list[int] = 1,
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

        factory_kwargs = {"device": device, "dtype": dtype, "bias": bias}

        self.layer_sizes = [input_size] + hidden_sizes
        self.num_layers = len(hidden_sizes)
        gh_layers = []
        align_layers = []
        for lidx, (ind, outd) in enumerate(
            zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        ):
            stride = strides[lidx]
            kernel = kernel_sizes[lidx]
            gh = torch.nn.Conv2d(
                ind,
                outd * 2,
                kernel_size=kernel,
                stride=stride,
                **factory_kwargs,
            )
            gh_layers.append(gh)
            align_layers.append(self._create_align_block(ind, outd, stride, kernel))
        self.gate_hidden_layers = torch.nn.ModuleList(gh_layers)
        if residual:
            self.align_layers = torch.nn.ModuleList(align_layers)

        self.dropout = max(min(dropout, 1.0), 0.0)
        self.residual = residual

    def _create_align_block(self, ind: int, outd: int, stride: int, kernel: int):
        if ind == outd & stride == 1:
            return torch.nn.Identity()
        else:
            return AlignConvBlock(ind, outd, kernel, stride)

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

        # if h is None:
        #     h = self.init_hidden_state(x)

        # # input to next layer
        # inp = x
        # next_hidden = []

        # # hidden states across layers
        # for lidx, gh_layer in enumerate(self.gate_hidden_layers):
        #     h_prev = h[lidx]

        #     out = mF.mingru(inp, h_prev, gh_layer.weight, gh_layer.bias)
        #     # (B,S,hidden_dims)

        #     # Save final hidden state of layer
        #     next_hidden.append(out[:, -1:])

        #     # Add skip connection
        #     if self.residual:
        #         # ModuleInterace is required to support dynamic indexing of
        #         # ModuleLists. See https://github.com/pytorch/pytorch/issues/47496
        #         al: ModuleInterface = self.align_layers[lidx]
        #         out = out + al.forward(inp)

        #     # Apply dropout (except for last)
        #     is_not_last = lidx < (self.num_layers - 1)
        #     if is_not_last and (self.dropout > 0):
        #         out = out * torch.bernoulli(
        #             torch.full_like(
        #                 out,
        #                 1 - self.dropout,
        #             )
        #         )

        #     # Next input is previous output
        #     inp = out

        # return out, next_hidden

    def init_hidden_state(self, x):
        return [
            mF.g(x.new_zeros(x.shape[0], 1, hidden_size))
            for hidden_size in self.layer_sizes[1:]
        ]


if __name__ == "__main__":
    rnn = MinGRU(input_size=3, hidden_sizes=[16, 32, 64], residual=True)
    x = torch.randn(10, 8, 3)
    h = rnn.init_hidden_state(x)

    out, hnext = rnn(x, h)
    print([hh.shape for hh in hnext])
    print(out.shape)

    scripted = torch.jit.script(rnn)
