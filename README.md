# torch-mingru
Torch MinGRU implementation based on 

> Feng, Leo, et al. "Were RNNs All We Needed?." (2024).

## Features
Currently the following features are implemented and tested:

 - **Parallel**: Efficient log-space parallel evaluation support plus sequential support for testing. Automatically dispatches to the most efficient implementation.
 - **Multilayer**: Stack multiple MinGRU layers via `num_layers=` arguments. When `num_layers>1`, the output hidden states of layer $i$ are passed as inputs to $i+1$.
 - **Dropout**: Via parameter `dropout=`, when > 0 all inputs of each layer are effected except for the last layer.
 - **Bias**: Biases in linear layers can be enabled and disabled via the `bias=` argument.
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
rnn = mingru.MinGRU(input_dims=3, hidden_dims=64, num_layers=2)

# Invoke without initial hidden state (will allocate zero hidden state)
h = rnn(torch.randn(10, 128, 3))
assert h.shape == (10, 128, 64)

# Invoke without initial hidden state and get all intermediate hidden states
h = rnn(torch.randn(10, 128, 3), return_all_outputs=True)
assert h.shape == (2, 10, 128, 64)

# Invoke with initial hidden state
h = rnn(torch.randn(10, 128, 3), h=torch.ones(10, 1, 64) * 0.5)
assert h.shape == (10, 128, 64)

# For sequential prediction use
data = torch.randn(10, 128, 3)
h0 = torch.ones(10, 1, 64) * 0.5
h = h0
h_seq = []
for i in range(128):
    # For more than 1 layers we need all intermediate hidden states
    # for next invocation
    h = rnn(data[:, i : i + 1], h, return_all_outputs=True)
    # However, we are usually interested in just the last one as output
    h_seq.append(h[-1])
h_seq = torch.cat(h_seq, 1)
# same as
h_par = rnn(data, h0)
assert torch.allclose(h_seq, h_par)

# Note, don't use all-zeros for initial hidden state, instead
# use the activation function
h = rnn(torch.randn(10, 128, 3), mingru.g(torch.zeros(10, 1, 64)))
assert h.shape == (10, 128, 64)
```

### Selective Copying
For a more complete example check the [selective_copying.py](./selective_copying.py) example, which attempts to learn to selectively pick specific tokens in order from a generated sequence.

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