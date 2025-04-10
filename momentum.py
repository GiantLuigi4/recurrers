import torch


# def momentum_transform(seq):
#     c = torch.zeros_like(seq)
#     d = torch.zeros_like(seq)
#     c[:, 0] = seq[:, 0]
#     d[:, 0] = seq[:, 0]
#
#     prev = torch.zeros_like(c[:, 0])
#     dprev = prev
#     dmot = prev
#     for n in range(0, seq.size(1)):
#         curr = seq[:, n]
#         dprev = curr - prev
#         dmot = (dmot + dprev) * 0.95
#         prev = curr
#         c[:, n] = (curr - dprev) + dmot
#         d[:, n] = dmot
#
#     return torch.complex(c, d)


@torch.jit.script
def graph(seq, dmot, prev, c, d, n: int):
    curr = seq[:, n]
    dprev = curr - prev
    dmot = (dmot + dprev) * 0.95
    c[:, n] = (curr - dprev) + dmot
    d[:, n] = dmot
    return dmot, curr


@torch._dynamo.disable
def no_graph(seq, c, d):
    prev = torch.zeros_like(c[:, 0])
    dmot = prev
    for n in range(0, seq.size(1)):
        dmot, prev = graph(seq, dmot, prev, c, d, n)


def momentum_transform(seq):
    c = torch.zeros_like(seq)
    d = torch.zeros_like(seq)
    c[:, 0] = seq[:, 0]
    d[:, 0] = seq[:, 0]
    no_graph(seq, c, d)
    return torch.complex(c, d)


# @torch.jit.script
# def imt(state):
#     c = torch.real(state)
#     d = torch.imag(state)
#
#     seq = torch.zeros_like(c)
#
#     seq[:, 0] = c[:, 0] / 0.95
#
#     for n in range(1, c.size(1)):
#         seq[:, n] = c[:, n] + (d[:, n] / 19.0) - d[:, n - 1]
#
#     return seq


@torch.jit.script
def imt(state):
    c = torch.real(state)
    d = torch.imag(state)
    seq = torch.zeros_like(c)
    seq[:, 0] = c[:, 0] + d[:, 0] / 19.0
    seq[:, 1:] = c[:, 1:] + (d[:, 1:] / 19.0) - d[:, :-1]
    return seq

# seq = torch.randn(1, 204800, 16)
# print(imt(momentum_transform(seq)) - seq)
