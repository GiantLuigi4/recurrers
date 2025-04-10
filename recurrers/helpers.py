import torch
import torch.nn as nn
import recurrers
import torch.nn.functional as F


class LightweightKnowledgeFeeder(recurrers.RecurrerLayer):
    """
        Utilizes a learnable knowledge bank to allow the model to learn "truthful" information
        Traditional transformers end up with this information encoded into the feed-forward networks
        This particular implementation is meant to be lightweight

        There is most likely a way to get one which can encode more information in the same amount of space
        However, when I was trying to make that, it ended up drastically increasing compute cost
    """

    def __init__(self, embed, lora, knowledge_bank, bank_size):
        super().__init__()
        self.lin_0 = nn.Linear(embed, lora)
        self.lin_1 = nn.Linear(lora, embed)
        self.lin_bank = nn.Linear(embed, knowledge_bank)
        self.lin_debank = nn.Linear(knowledge_bank, embed)
        self.bank = nn.Parameter(
            torch.rand(bank_size, knowledge_bank),
            requires_grad=True
        )

    def make_state(self, batch_size: int):
        return []

    def recurrent_forward(self, x: torch.Tensor, *state):
        return self.forward(x)

    def forward(self, x: torch.Tensor, *state):
        x = x.reshape(x.size(0), -1, x.size(-1))
        lora = F.gelu(self.lin_0(x))
        x = x + self.lin_1(lora)
        y = self.lin_bank(x)
        bank_scores = y @ self.bank.T
        bank_weights = F.softmax(bank_scores, dim=-1)
        output = bank_weights.unsqueeze(-1) * self.bank.view(1, 1, *self.bank.shape)
        output = output.sum(dim=-2)
        x = x + self.lin_debank(output)
        return x, []

# TODO: sequence recurrer (nn.Sequential for recurrers)
