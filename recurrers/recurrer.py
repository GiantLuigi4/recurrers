import torch
import torch.nn as nn


class RecurrerLayer(nn.Module):
    def __init__(self):
        super().__init__()

    """
        Creates the initial state for the module
    """

    def make_state(self, batch_size: int):
        raise "make_state must be implemented"

    """
        Heavily recommended to override this
        Recurrently calls forward
    """

    # "optimizing" this with triton obliterates performance
    # unfortunately, if you want to be able to have efficient recurrent forwards, you have to override this
    # unless triton makes cases specifically for handling loops like this, I cannot do much about that
    @torch._dynamo.disable
    def recurrent_forward(self, x: torch.Tensor, *state):
        buf = []  # the output size is unknown by this base function, so a list is used
        for i in range(0, x.size(1)):
            v, *state = self.forward(x[:, i, :], *state)
            buf.append(v)
        return torch.cat(buf, 1), *state

    """
        Processes a single new piece of information
    """

    def forward(self, x: torch.Tensor, *state):
        raise "forward must be implemented"

    # for the majority of layers, there is no compute gating grid
    """
        The compute grid evaluates where the evaluation must break
        There are two modes of breaking: forwards breaking and bidirectional breaking
        With forwards breaking, the lower layers can continue to evaluate all the way through
        With bidirectional breaking, both directions have to break evaluation until the layer reaches the break point
        This is useful for layers that affect the state of past or future layers
    """

    @torch._dynamo.disable
    def compute_grid(self, *state):
        return []


class ModelState:
    def __init__(self):
        self.states: list = []

    def make_state(self, mdl, batch_size: int):
        for lyr in mdl.layers:
            self.states.append(lyr.make_state(batch_size))


"""
Recurrers: "efficient" and "simple" custom-made recurrence models
The goal for recurrers is to make it simple to make advanced recurrence based models
and have them run with high efficiency via triton
"""


class Recurrer(nn.Module):
    layers: nn.ModuleList

    def __init__(self, layers: list):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    # @torch._dynamo.disable
    def forward(self, x: torch.Tensor, state: ModelState):
        gate_pos = 0

        buf = []

        while gate_pos < x.size(1):
            gate_grid = x.size(1)

            for i in range(0, len(self.layers)):
                grid = self.layers[i].compute_grid(state.states[i])
                # TODO: support for front-only locking
                #       with front-only locking, lower layers can carry all the way through, but higher layers must lock
                for e in grid:
                    gate_grid = min(gate_grid, e['lock'])

            gate_seq = x[:, gate_pos:(gate_pos + gate_grid), :]

            for i in range(0, len(self.layers)):
                clayer = self.layers[i]
                cstate = state.states[i]
                gate_seq, *cstate = clayer.recurrent_forward(gate_seq, *cstate)
                state.states[i] = cstate

            gate_pos = gate_pos + gate_grid
            buf.append(gate_seq)
        return torch.cat(buf, dim=1), state

    def make_state(self, batch_size: int = 1):
        state = ModelState()
        state.make_state(self, batch_size)
        return state
