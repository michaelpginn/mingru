import torch

import mingru


@torch.no_grad()
def test_mingru_interface():
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
        out, h = rnn(x[:, i : i + 1], h=h)  # NOQA
        out_seq.append(out)
    out_seq = torch.cat(out_seq, 1)
    assert out_seq.shape == (B, S, 64)

    # Parallel prediction pattern
    out_par, h = rnn(x, rnn.init_hidden_state(x))
    assert torch.allclose(out_seq, out_par, atol=1e-4)


@torch.no_grad()
def test_mingru_conv_interface():
    # Instantiate
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
        out, h = rnn(x[:, i : i + 1], h=h)  # NOQA
        out_seq.append(out)
    out_seq = torch.cat(out_seq, 1)
    assert out_seq.shape == (B, S, 64, 8, 8)

    # Parallel prediction pattern
    out_par, h = rnn(x, rnn.init_hidden_state(x))
    assert torch.allclose(out_seq, out_par, atol=1e-4)
