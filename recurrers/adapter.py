import torch
import torch.nn as nn
import recurrers


class GRUStateGen(nn.Module):
    def __init__(self, embed, layers=1):
        super().__init__()
        self.state = nn.Parameter(torch.rand(layers, 1, embed))

    def forward(self, batch: int):
        return (self.state.repeat(1, batch, 1),)


class LSTMStateGen(nn.Module):
    def __init__(self, embed, layers=1):
        super().__init__()
        self.state = nn.Parameter(torch.rand(2, layers, 1, embed))

    def forward(self, batch: int):
        return (tuple(self.state.repeat(1, 1, batch, 1)),)


class RNNAdapter(recurrers.RecurrerLayer):
    """
        Adapter is intended for batch first rnns
    """

    def __init__(self, initial_state, parent, proj_out=None):
        super().__init__()
        self.initial_state = initial_state
        self.layer = parent
        self.proj_out = proj_out
        self.recurrent_forward = self.forward

    def make_state(self, batch_size: int):
        return self.initial_state(batch_size)

    def parallel_forward(self, x: torch.Tensor, *state):
        xvv, state = self.layer(x, *state)
        if self.proj_out is not None:
            return self.proj_out(xvv[0]), xvv[1]
        return xvv, state

    def forward(self, x: torch.Tensor, *state):
        xvv, state = self.layer(x.view(-1, 1, x.shape[-1]), *state)
        if self.proj_out is not None:
            return self.proj_out(xvv[0]), xvv[1]
        return xvv, state


class MultiheadAttention(recurrers.RecurrerLayer):
    """
        Adapter is intended for batch first rnns
    """

    def __init__(self, embed, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed, n_heads,
            batch_first=True
        )

    def make_state(self, batch_size: int):
        # TODO: more efficient recurrence via activation caching
        return []

    def forward(self, x: torch.Tensor, *state):
        attn_mask = torch.triu(torch.ones([x.shape[1], x.shape[1]], device=x.device), diagonal=1).bool()
        xvv = self.attn(
            x, x, x, attn_mask=attn_mask
        )
        return xvv

    def parallel_forward(self, x: torch.Tensor, *state):
        attn_mask = torch.triu(torch.ones([x.shape[1], x.shape[1]], device=x.device), diagonal=1).bool()
        xvv = self.attn(
            x, x, x, attn_mask=attn_mask
        )
        return xvv


class FeedNetAdapter(recurrers.RecurrerLayer):
    """
        Adapter is intended for batch first rnns
    """

    def __init__(self, mdl):
        super().__init__()
        self.mdl = mdl
        self.recurrent_forward = self.forward

    def make_state(self, batch_size: int):
        return ()

    def parallel_forward(self, x: torch.Tensor, *state):
        return self.mdl(x), *state

    def forward(self, x: torch.Tensor, *state):
        return self.mdl(x), *state
