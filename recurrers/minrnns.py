import torch
import torch.nn as nn
import recurrers
from minGRU_pytorch.minLSTM import minLSTM
from minGRU_pytorch.minGRU import minGRU


class MinState(nn.Module):
    def __init__(self, embed, layers=1):
        super().__init__()
        self.state = nn.Parameter(torch.rand(layers, 1, embed))

    def forward(self, batch: int):
        return (self.state.repeat(1, batch, 1),)


"""
Recurrer wrapper for https://github.com/lucidrains/minGRU-pytorch minLSTM
"""


class MinLSTM(recurrers.RecurrerLayer):
    """
        Adapter is intended for batch first rnns
    """

    def __init__(self, initial_state, input_shape, units):
        super().__init__()
        self.initial_state = initial_state
        self.input_shape = input_shape
        self.units = units

        self.layer = minLSTM(
            input_shape, units / input_shape
        )
        self.proj_out = None

    def make_state(self, batch_size: int):
        return self.initial_state(batch_size)

    def parallel_forward(self, x: torch.Tensor, *state):
        xvv, state = self.layer(x, *state, True)
        if self.proj_out is not None:
            return self.proj_out(xvv[0]), xvv[1]
        return xvv, state

    def forward(self, x: torch.Tensor, *state):
        xvv, state = self.layer(x, *state, True)
        if self.proj_out is not None:
            return self.proj_out(xvv[0]), xvv[1]
        return xvv, state


"""
Recurrer wrapper for https://github.com/lucidrains/minGRU-pytorch minGRU
"""


class MinGRU(recurrers.RecurrerLayer):
    """
        Adapter is intended for batch first rnns
    """

    def __init__(self, initial_state, input_shape, units):
        super().__init__()
        self.initial_state = initial_state
        self.input_shape = input_shape
        self.units = units

        self.layer = minGRU(
            input_shape, units / input_shape
        )
        self.proj_out = None

    def make_state(self, batch_size: int):
        return self.initial_state(batch_size)

    def parallel_forward(self, x: torch.Tensor, *state):
        xvv, state = self.layer(x, *state, True)
        if self.proj_out is not None:
            return self.proj_out(xvv[0]), xvv[1]
        return xvv, state

    def forward(self, x: torch.Tensor, *state):
        xvv, state = self.layer(x, *state, True)
        if self.proj_out is not None:
            return self.proj_out(xvv[0]), xvv[1]
        return xvv, state
