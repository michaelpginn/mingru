import pytest
import torch

import mingru


def test_mingru_basic():

    rnn = mingru.MinGRU(input_dims=1, hidden_dims=5, num_layers=2)

    h0 = torch.zeros(1, 1, 5)
    x = torch.randn(1, 5, 1)

    h = mingru.functional.g(h0)  # <- don't forget the activation
    out_seq = []
    for i in range(5):
        out, h = rnn(x[:, i : i + 1], h)
        assert h.shape == (2, 1, 1, 5)
        out_seq.append(out)
    out_seq = torch.cat(out_seq, 1)

    out_par, h = rnn(x, mingru.functional.g(h0))
    assert out_par.shape == (1, 5, 5)
    assert torch.allclose(out_seq, out_par)


@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_mingru(num_layers):

    rnn = mingru.MinGRU(input_dims=3, hidden_dims=5, num_layers=num_layers)

    h0 = torch.zeros(2, 1, 5)
    x = torch.randn(2, 5, 3)

    # sequential pattern
    h = mingru.functional.g(h0)
    out_seq = []
    for i in range(x.shape[1]):
        # For more than 1 layers we need all intermediate hidden states
        # for next invocation
        out, h = rnn(x[:, i : i + 1], h)
        # However, we are usually interested in just the last one as output
        out_seq.append(out)
    out_seq = torch.cat(out_seq, 1)

    out_par, h = rnn(x, mingru.functional.g(h0))
    assert out_par.shape == (2, 5, 5)
    assert h.shape == (num_layers, 2, 1, 5)
    assert torch.allclose(out_seq, out_par)


@torch.no_grad()
def test_interface():

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
    out_hs = []
    for i in range(data.shape[1]):
        out, h = rnn(data[:, i : i + 1], h=h)
        out_hs.append(h)
        out_seq.append(out)
    out_seq = torch.cat(out_seq, 1)
    assert out_seq.shape == (B, S, H)

    # Parallel prediction pattern
    out_par, h = rnn(data, h0)
    assert torch.allclose(out_seq, out_par, atol=1e-4)

    # Note, don't use <= 0 for initial hidden state, instead
    h0 = mingru.functional.g(torch.zeros(10, 1, 64))
