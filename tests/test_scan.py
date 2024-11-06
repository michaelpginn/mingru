import torch

from mingru import parallel_scan_log


def test_parallel_scan():
    a = torch.rand(10) + 0.1
    b = torch.rand(10) + 0.1
    x0 = torch.rand(1) + 0.1

    xt_seq = [x0]
    for i in range(10):
        xt_seq.append(a[i] * xt_seq[-1] + b[i])
    xt_seq = torch.tensor(xt_seq[1:])

    xt_par = parallel_scan_log(
        torch.log(a).view(1, -1, 1),
        torch.cat([torch.log(x0), torch.log(b)], 0).view(1, -1, 1),
    )[:, 1:, :].view(-1)

    assert torch.allclose(xt_seq, xt_par)
