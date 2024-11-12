# torch-mingru
Torch MinGRU implementation based on 

> Feng, Leo, et al. "Were RNNs All We Needed?." (2024).

## Features
Currently the following features are implemented and tested:

 - **Parallel**: Efficient log-space parallel evaluation support plus sequential support for testing. Automatically dispatches to the most efficient implementation.
 - **Multilayer**: Stack multiple MinGRU layers via `num_layers=` arguments. When `num_layers>1`, the output hidden states of layer $i$ are passed as inputs to $i+1$.
 - **Dropout**: Via parameter `dropout=`, when > 0 all inputs of each layer are effected except for the last layer.
 - **Bias**: Biases in linear layers can be enabled and disabled via the `bias=` argument.
 - **Residuals**: Residual connections betweeen outputs of minGRU layers via `residual=` argument.
 - **Scripting**: MinGRU is compatible with `torch.jit.script`.
 - **Compatibility**: Interface of *mingru* is mostly compatible with that of `torch.nn.GRU`, except that bi-directional and sequence-first arguments are not supported.

## Installation

```shell
# Install directly from github
pip install git+https://github.com/cheind/mingru.git
```

## Usage

```python
import torch
import mingru

# Instantiate
B, I, H, L, S = 10, 3, 64, 2, 128
rnn = mingru.MinGRU(
    input_dims=I,
    hidden_dims=H,
    num_layers=L,
    dropout=0.0,
    residual=True,
).eval()

# Invoke for input x with sequence length 128 and batch-size 10
# This will allocate 'zero' hidden state for each layer.
out, h = rnn(torch.randn(B, S, I))
assert out.shape == (B, S, H)
assert h.shape == (L, B, 1, H)

# Invoke with initial/previous hidden state.
# Hidden state must be expandable to shape (L,B,1,H)
out, h = rnn(torch.randn(B, S, I), h=torch.ones(B, 1, H) * 0.5)
assert out.shape == (B, S, H)
assert h.shape == (L, B, 1, H)

# Sequential prediction pattern
data = torch.randn(B, S, I)
h0 = torch.ones(L, B, 1, H) * 0.5
h = h0
out_seq = []
for i in range(data.shape[1]):
    out, h = rnn(data[:, i : i + 1], h=h)
    out_seq.append(out)
out_seq = torch.cat(out_seq, 1)
assert out_seq.shape == (B, S, H)

# Parallel prediction pattern
out_par, h = rnn(data, h0)
assert torch.allclose(out_seq, out_par, atol=1e-4)

# Note, don't use <= 0 for initial hidden state, instead
h0 = mingru.functional.g(torch.zeros(10, 1, 64))
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
