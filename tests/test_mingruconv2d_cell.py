import torch
import io

import mingru


def test_mingruconv2dcell():

    rnn = mingru.MinConv2dGRUCell(
        input_size=3,
        hidden_size=5,
        kernel_size=3,
        stride=1,
        padding=1,
    )

    x = torch.randn(2, 5, 3, 32, 32)
    h = rnn.init_hidden_state(x)
    assert h.shape == (2, 1, 5, 32, 32)

    out_seq = []
    for i in range(x.shape[1]):
        h = rnn(x[:, i : i + 1], h)
        assert h.shape == (2, 1, 5, 32, 32)
        out_seq.append(h)
    out_seq = torch.cat(out_seq, 1)

    out_par = rnn(x, rnn.init_hidden_state(x))
    assert out_par.shape == (2, 5, 5, 32, 32)
    assert torch.allclose(out_seq, out_par)


def test_mingruconv2dcell_downsample():
    rnn = mingru.MinConv2dGRUCell(
        input_size=3,
        hidden_size=5,
        kernel_size=3,
        stride=2,
        padding=1,
    )

    x = torch.randn(2, 5, 3, 32, 32)
    h = rnn.init_hidden_state(x)
    assert h.shape == (2, 1, 5, 16, 16)

    out_seq = []
    for i in range(x.shape[1]):
        h = rnn(x[:, i : i + 1], h)
        out_seq.append(h)
    out_seq = torch.cat(out_seq, 1)

    out_par = rnn(x, rnn.init_hidden_state(x))
    assert torch.allclose(out_seq, out_par)


def test_mingrucell_scripting():
    rnn = mingru.MinConv2dGRUCell(
        input_size=3,
        hidden_size=5,
        kernel_size=3,
        stride=2,
        padding=1,
    )

    x = torch.randn(2, 5, 3, 32, 32)
    h = rnn.init_hidden_state(x)

    rnn_out = rnn(x, h)

    scripted = torch.jit.script(rnn)
    scripted_out = scripted(x, h)

    assert torch.allclose(scripted_out, rnn_out, atol=1e-4)

    scripted_out = scripted(x)
    rnn_out = rnn(x)

    assert torch.allclose(scripted_out, rnn_out, atol=1e-4)

    # Save load
    buffer = io.BytesIO()
    torch.jit.save(scripted, buffer)

    buffer.seek(0)
    loaded = torch.jit.load(buffer, map_location=torch.device("cpu"))
    loaded_out = loaded(x)

    rnn_out = rnn(x)
    assert torch.allclose(loaded_out, rnn_out, atol=1e-4)
