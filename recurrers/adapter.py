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

    def forward(self, x: torch.Tensor, *state):
        xvv = self.layer(x, *state)
        if self.proj_out is not None:
            return self.proj_out(xvv[0]), xvv[1]
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

    def recurrent_forward(self, x: torch.Tensor, *state):
        return self.mdl(x), *state

    def forward(self, x: torch.Tensor, *state):
        return self.mdl(x), *state
