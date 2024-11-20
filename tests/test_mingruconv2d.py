import pytest
import torch
import io

import mingru


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("residual", [True, False])
def test_mingruconv2d(bias, residual):

    rnn = mingru.MinConv2dGRU(
        input_size=1,
        hidden_sizes=[3, 5],
        kernel_sizes=[3, 3],
        paddings=[1, 1],
        strides=1,
        bias=bias,
        residual=residual,
    )

    assert rnn.layer_sizes == (1, 3, 5)
    assert rnn.num_layers == 2

    x = torch.randn(2, 5, 1, 32, 32)
    h = rnn.init_hidden_state(x)
    assert h[0].shape == (2, 1, 3, 32, 32)
    assert h[1].shape == (2, 1, 5, 32, 32)

    out_seq = []
    for i in range(x.shape[1]):
        out, h = rnn(x[:, i : i + 1], h)
        out_seq.append(out)
    out_seq = torch.cat(out_seq, 1)

    out_par, h = rnn(x, rnn.init_hidden_state(x))

    assert torch.allclose(out_seq, out_par, atol=1e-4)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("residual", [True, False])
def test_mingru_scripting(bias, residual):
    rnn = mingru.MinConv2dGRU(
        input_size=2,
        hidden_sizes=[3, 5],
        kernel_sizes=[3, 3],
        paddings=[1, 1],
        strides=1,
        bias=bias,
        residual=residual,
    )

    x = torch.randn(2, 5, 2, 32, 32)
    h = rnn.init_hidden_state(x)
    rnn_out, rnn_h = rnn(x, h)

    scripted = torch.jit.script(rnn)
    scripted_out, scripted_h = scripted(x, h)

    assert torch.allclose(scripted_out, rnn_out, atol=1e-4)
    assert torch.allclose(scripted_h[0], rnn_h[0], atol=1e-4)
    assert torch.allclose(scripted_h[1], rnn_h[1], atol=1e-4)

    scripted_out, scripted_h = scripted(x)
    rnn_out, rnn_h = rnn(x)

    assert torch.allclose(scripted_out, rnn_out, atol=1e-4)
    assert torch.allclose(scripted_h[0], rnn_h[0], atol=1e-4)
    assert torch.allclose(scripted_h[1], rnn_h[1], atol=1e-4)

    # Save load
    buffer = io.BytesIO()
    torch.jit.save(scripted, buffer)

    buffer.seek(0)
    loaded = torch.jit.load(buffer, map_location=torch.device("cpu"))
    loaded_out, loaded_h = loaded(x)

    rnn_out, rnn_h = rnn(x)
    assert torch.allclose(loaded_out, rnn_out, atol=1e-4)
    assert torch.allclose(loaded_h[0], rnn_h[0], atol=1e-4)
    assert torch.allclose(loaded_h[1], rnn_h[1], atol=1e-4)


# @pytest.mark.parametrize("bias", [True, False])
# @pytest.mark.parametrize("residual", [True, False])
# def test_mingru_scripting(bias, residual):
#     rnn = mingru.MinGRU(
#         input_size=2,
#         hidden_sizes=[3, 5],
#         bias=bias,
#         residual=residual,
#     )

#     assert rnn.layer_sizes == (2, 3, 5)
#     assert rnn.num_layers == 2

#     x = torch.randn(1, 128, 2)
#     h = rnn.init_hidden_state(x)
#     rnn_out, rnn_h = rnn(x, h)

#     scripted = torch.jit.script(rnn)
#     scripted_out, scripted_h = scripted(x, h)

#     assert torch.allclose(scripted_out, rnn_out, atol=1e-4)
#     assert torch.allclose(scripted_h[0], rnn_h[0], atol=1e-4)
#     assert torch.allclose(scripted_h[1], rnn_h[1], atol=1e-4)

#     scripted_out, scripted_h = scripted(x)
#     rnn_out, rnn_h = rnn(x)

#     assert torch.allclose(scripted_out, rnn_out, atol=1e-4)
#     assert torch.allclose(scripted_h[0], rnn_h[0], atol=1e-4)
#     assert torch.allclose(scripted_h[1], rnn_h[1], atol=1e-4)

#     # Save load
#     buffer = io.BytesIO()
#     torch.jit.save(scripted, buffer)

#     buffer.seek(0)
#     loaded = torch.jit.load(buffer, map_location=torch.device("cpu"))
#     loaded_out, loaded_h = loaded(x)

#     rnn_out, rnn_h = rnn(x)
#     assert torch.allclose(loaded_out, rnn_out, atol=1e-4)
#     assert torch.allclose(loaded_h[0], rnn_h[0], atol=1e-4)
#     assert torch.allclose(loaded_h[1], rnn_h[1], atol=1e-4)


# # @pytest.mark.parametrize("num_layers", [1, 2, 3])
# # def test_mingru(num_layers):

# #     rnn = mingru.MinGRU(input_dims=3, hidden_dims=5, num_layers=num_layers)

# #     h0 = torch.zeros(2, 1, 5)
# #     x = torch.randn(2, 5, 3)

# #     # sequential pattern
# #     h = mingru.functional.g(h0)
# #     out_seq = []
# #     for i in range(x.shape[1]):
# #         # For more than 1 layers we need all intermediate hidden states
# #         # for next invocation
# #         out, h = rnn(x[:, i : i + 1], h)
# #         # However, we are usually interested in just the last one as output
# #         out_seq.append(out)
# #     out_seq = torch.cat(out_seq, 1)

# #     out_par, h = rnn(x, mingru.functional.g(h0))
# #     assert out_par.shape == (2, 5, 5)
# #     assert h.shape == (num_layers, 2, 1, 5)
# #     assert torch.allclose(out_seq, out_par)
