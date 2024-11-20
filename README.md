# torch-mingru
PyTorch (convolutional) MinGRU implementation based on 

> Feng, Leo, et al. "Were RNNs All We Needed?" (2024).

Convolutional MinGRU based on

> Heindl, Christoph et al. "Convolutional MinGRU" (2024).

## Features
In alignment with torch recurrent modules, **mingru** provides the following core modules
 - `mingru.MinGRUCell` single layer MinGRU
 - `mingru.MinGRU` multi-layer stacked MinGRU 
 - `mingru.MinConv2dGRUCell` single layer convolutional MinGRU
 - `mingru.MinConv2dGRU` multi-layer stacked convolutional MinGRU

Each module supports the following features (if applicable to type)
 - **Parallel**: Efficient log-space parallel evaluation support plus sequential support for testing. Automatically dispatches to the most efficient implementation.
 - **Multilayer**: Stack multiple MinGRU layers via `hidden_sizes=` arguments. When `len(hidden_sizes)>1`, the output hidden states of layer $i$ are passed as inputs to $i+1$. Varying hidden sizes are supported.
 - **Dropout**: Via parameter `dropout=`, when > 0 all inputs of each layer are effected except for the last layer.
 - **Residual**: Residual connections betweeen outputs of minGRU layers via `residual=` argument.
 - **Bias**: Biases in linear layers can be enabled and disabled via the `bias=` argument.
 - **Scripting**: MinGRU is compatible with `torch.jit.script`.
 - **Compatibility**: Interface of `mingru.*` is mostly compatible with that of `torch.nn.GRU/GRUCell`, except that bi-directional and sequence-first arguments are not supported. Cells in **mingru** also support sequence arguments to benefit from parallel computation.

## Installation

```shell
# Install directly from github
pip install git+https://github.com/cheind/mingru.git
```

## Usage

### MinGRU

The following snippet demonstrates a multi-layer stacked MinGRU.

```python
import torch
import mingru

# Instantiate
B, input_size, hidden_sizes, S = 10, 3, [32, 64], 128
rnn = mingru.MinGRU(
    input_size=input_size,
    hidden_sizes=hidden_sizes,
    dropout=0.0,
    residual=True,
).eval()

# Invoke for input x with sequence length S and batch-size B
# This will implicitly assume a 'zero' hidden state
# for each layer.
x = torch.randn(B, S, input_size)
out, h = rnn(x)
assert out.shape == (B, S, 64)
assert h[0].shape == (B, 1, 32)
assert h[1].shape == (B, 1, 64)

# Invoke with initial/previous hidden states.
h = rnn.init_hidden_state(x)
out, h = rnn(torch.randn(B, S, input_size), h=h)

# Sequential prediction pattern
h = rnn.init_hidden_state(x)
out_seq = []
for i in range(x.shape[1]):
    out, h = rnn(x[:, i : i + 1], h=h)
    out_seq.append(out)
out_seq = torch.cat(out_seq, 1)
assert out_seq.shape == (B, S, 64)

# Parallel prediction pattern
out_par, h = rnn(x, rnn.init_hidden_state(x))
assert torch.allclose(out_seq, out_par, atol=1e-4)
```

### MinConv2dGRU

Following sample demonstrates convolutional multi-layer stacked MinGRUs.


```python
import torch
import mingru

B, S = 5, 10
input_size = 3
hidden_sizes = [16, 32, 64]
kernel_sizes = [3, 3, 3]
padding = 1
stride = 2

rnn = mingru.MinConv2dGRU(
    input_size=input_size,
    hidden_sizes=hidden_sizes,
    kernel_sizes=kernel_sizes,
    paddings=padding,
    strides=stride,
    dropout=0.0,
    residual=True,
).eval()

# Invoke for input x with sequence length S and batch-size B
# This will implicitly assume a 'zero' hidden state
# for each layer.
x = torch.randn(B, S, input_size, 64, 64)
out, h = rnn(x)
assert out.shape == (B, S, 64, 8, 8)
assert h[0].shape == (B, 1, 16, 32, 32)
assert h[1].shape == (B, 1, 32, 16, 16)
assert h[2].shape == (B, 1, 64, 8, 8)

# Invoke with initial/previous hidden states.
h = rnn.init_hidden_state(x)
out, h = rnn(x, h=h)

# Sequential prediction pattern
h = rnn.init_hidden_state(x)
out_seq = []
for i in range(x.shape[1]):
    out, h = rnn(x[:, i : i + 1], h=h)
    out_seq.append(out)
out_seq = torch.cat(out_seq, 1)
assert out_seq.shape == (B, S, 64, 8, 8)

# Parallel prediction pattern
out_par, h = rnn(x, rnn.init_hidden_state(x))
assert torch.allclose(out_seq, out_par, atol=1e-4)
```

### Selective Copying
For a more complete example check the [selective_copying.py](./selective_copying.py), which attempts to learn to selectively pick specific tokens in order from a generated sequence.

```shell
python selective_copying.py
    ...
    Step [1381/1500], Loss: 0.0005, Accuracy: 98.44%
    Step [1401/1500], Loss: 0.0002, Accuracy: 99.61%
    Step [1421/1500], Loss: 0.0005, Accuracy: 97.66%
    Step [1441/1500], Loss: 0.0009, Accuracy: 98.05%
    Step [1461/1500], Loss: 0.0014, Accuracy: 96.88%
    Step [1481/1500], Loss: 0.0005, Accuracy: 98.05%
    Validation Accuracy: 98.44%
```

Per default, the example is configured for a small usecase (sequence length 64, vocab size 6, memorize 4), but you might just change to a much larger test by adopting `cfg` dict at the end of the file.

Task is based on
> Gu, Albert, and Tri Dao. "Mamba: Linear-time sequence modeling with selective state spaces." (2023).
