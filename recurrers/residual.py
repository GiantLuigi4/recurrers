import torch
import torch.nn as nn
import recurrers


class Mix(recurrers.RecurrerLayer):
    """
        A residual layer which uses learnable biases towards pre and post
    """

    def __init__(self, module: recurrers.RecurrerLayer, weight: float = 0.1):
        super().__init__()
        self.bias_in = nn.Parameter(torch.tensor(1 - weight), requires_grad=True)
        self.bias_out = nn.Parameter(torch.tensor(weight), requires_grad=True)
        self.module = module

    def parallel_forward(self, x: torch.Tensor, *state):
        mv = self.module.parallel_forward(x, *state)
        return x * self.bias_in + mv[0] * self.bias_out, mv[1]

    def forward(self, x: torch.Tensor, *state):
        mv = self.module.forward(x, *state)
        return x * self.bias_in + mv[0] * self.bias_out, mv[1]

    def compute_grid(self, *state):
        return self.module.compute_grid(*state)

    def make_state(self, batch_size: int):
        return self.module.make_state(batch_size)


class OrganisedResidual(recurrers.RecurrerLayer):
    """
        A variant of mix which uses a linear layer to reorganize and scale information
    """

    def __init__(self, embedding: int, module: recurrers.RecurrerLayer):
        super().__init__()
        self.lin_in = nn.Linear(embedding, embedding)
        self.lin_out = nn.Linear(embedding, embedding)
        self.module = module

    def parallel_forward(self, x: torch.Tensor, *state):
        mv = self.module.parallel_forward(x, *state)
        return self.lin_in(x) + self.lin_out(mv[0]), mv[1:]

    def forward(self, x: torch.Tensor, *state):
        mv = self.module.forward(x, *state)
        return self.lin_in(x) + self.lin_out(mv[0]), mv[1:]

    def compute_grid(self, *state):
        return self.module.compute_grid(*state)

    def make_state(self, batch_size: int):
        return self.module.make_state(batch_size)


class GatedResidual(recurrers.RecurrerLayer):
    """
        Concatenates input and output, blends them together using a linear
    """

    def __init__(self, embedding: int, module: recurrers.RecurrerLayer):
        super().__init__()
        self.lin = nn.Linear(embedding * 2, embedding)
        self.module = module

    def parallel_forward(self, x: torch.Tensor, *state):
        mv = self.module.parallel_forward(x, *state)
        return self.lin(torch.cat((x, mv[0]), dim=-1)), mv[1:]

    def forward(self, x: torch.Tensor, *state):
        mv = self.module.forward(x, *state)
        return self.lin(torch.cat((x, mv[0]), dim=-1)), mv[1:]

    def compute_grid(self, *state):
        return self.module.compute_grid(*state)

    def make_state(self, batch_size: int):
        return self.module.make_state(batch_size)


class FFResidual(recurrers.RecurrerLayer):
    """
        Residual using a feed forward network
    """

    def __init__(self, embedding: int, module: recurrers.RecurrerLayer):
        super().__init__()
        self.lin_in = nn.Linear(embedding * 2, embedding * 4)
        self.activ = nn.GELU()
        self.lin_out = nn.Linear(embedding * 4, embedding)
        self.module = module

    def parallel_forward(self, x: torch.Tensor, *state):
        mv = self.module.parallel_forward(x, *state)
        activ = self.activ(self.lin_in(torch.cat((x, mv[0]), dim=-1)))
        return self.lin_out(activ), mv[1:]

    def forward(self, x: torch.Tensor, *state):
        mv = self.module.forward(x, *state)
        activ = self.activ(self.lin_in(torch.cat((x, mv[0]), dim=-1)))
        return self.lin_out(activ), mv[1:]

    def compute_grid(self, *state):
        return self.module.compute_grid(*state)

    def make_state(self, batch_size: int):
        return self.module.make_state(batch_size)
