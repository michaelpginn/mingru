import torch

import mingru


def test_conv_mingru_functional():

    x = torch.randn(1, 10, 2, 32, 32)
    h0 = torch.zeros(1, 1, 5, 32, 32)

    kernel = torch.randn(5 * 2, 2, 3, 3)
    bias = torch.randn(5 * 2)

    h = mingru.functional.g(h0)  # <- don't forget the activation

    out_seq = []
    for i in range(x.shape[1]):
        # For more than 1 layers we need all intermediate hidden states
        # for next invocation
        h = mingru.functional.mingru(x[:, i : i + 1], h, kernel, bias)
        out_seq.append(h)
    out_seq = torch.cat(out_seq, 1)

    h = mingru.functional.g(h0)
    out_par = mingru.functional.mingru(x, h, kernel, bias)

    assert torch.allclose(out_par, out_seq, atol=1e-4)


def test_mingru_basic():

    rnn = mingru.MinConv2dGRU(input_dims=1, hidden_dims=5, num_layers=2)

    h0 = torch.zeros(1, 1, 5, 16, 16)
    x = torch.randn(1, 5, 1, 16, 16)

    h = mingru.functional.g(h0)  # <- don't forget the activation
    out_seq = []
    for i in range(5):
        out, h = rnn(x[:, i : i + 1], h)
        assert h.shape == (2, 1, 1, 5, 16, 16)
        out_seq.append(out)
    out_seq = torch.cat(out_seq, 1)

    out_par, h = rnn(x, mingru.functional.g(h0))
    assert out_par.shape == (1, 5, 5, 16, 16)
    assert torch.allclose(out_seq, out_par)


@torch.no_grad()
def test_conv_interface():

    # Instantiate

    B, input_dims, hidden_dims, L, S, H, W = 10, 3, 64, 2, 128, 16, 16

    rnn = mingru.MinConv2dGRU(
        input_dims=input_dims,
        hidden_dims=hidden_dims,
        num_layers=L,
        dropout=0.0,
        residual=True,
        kernel_size=3,
    ).eval()

    # Invoke for input x with sequence length 128 and batch-size 10
    # This will allocate 'zero' hidden state for each layer.
    out, h = rnn(torch.randn(B, S, input_dims, H, W))
    assert out.shape == (B, S, hidden_dims, H, W)
    assert h.shape == (L, B, 1, hidden_dims, H, W)

    # Sequential prediction pattern
    data = torch.randn(B, S, input_dims, H, W)
    h0 = torch.ones(L, B, 1, hidden_dims, H, W) * 0.5
    h = h0
    out_seq = []
    for i in range(data.shape[1]):
        out, h = rnn(data[:, i : i + 1], h=h)
        out_seq.append(out)
    out_seq = torch.cat(out_seq, 1)
    assert out_seq.shape == (B, S, hidden_dims, H, W)

    # Parallel prediction pattern
    out_par, h = rnn(data, h0)
    assert torch.allclose(out_seq, out_par, atol=1e-4)
