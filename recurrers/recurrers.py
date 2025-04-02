import torch
import torch.nn as nn
import recurrers


class Sobloid(recurrers.RecurrerLayer):
    def __init__(self, embed):
        super().__init__()
        self.lin_in = nn.Linear(embed, embed)
        self.lin_dif = nn.Linear(embed, embed)
        self.norm = nn.RMSNorm(embed)
        self.lin_intermediate = nn.Linear(embed, embed)
        self.lin_final = nn.Linear(embed, embed)
        self.embed = embed

    def make_state(self, batch_size: int):
        return (
            torch.zeros(batch_size, 1, self.embed).to(self.lin_in.weight.device),
            0,
            torch.zeros(batch_size, 1, self.embed).to(self.lin_in.weight.device)
        )

    def recurrent_forward(self, x: torch.Tensor, state: torch.Tensor, index: int, state_cvv: torch.Tensor):
        x = self.lin_in(x)

        cvv: torch.Tensor = state_cvv

        xE: torch.Tensor = torch.cat([x[:, 0:1, :] - state, torch.diff(x, dim=1)], dim=1).contiguous()
        res: torch.Tensor = torch.zeros_like(xE, device=x.device)

        xE: torch.Tensor = self.lin_dif(xE)

        # this portion may be optimized
        def graphable(cvv, i, index):
            cvv = cvv + xE[:, i, :]
            res[:, i, :] = cvv

            if index % 32 == 0:
                cvv = self.norm(cvv)
                cvv = self.lin_intermediate(cvv)

            return cvv

        # "optimizing" this with triton obliterates performance
        @torch._dynamo.disable
        def no_graph(x, cvv, index):
            for i in range(0, x.size(1)):
                cvv = graphable(cvv, i, index)
                index = index + 1
            return cvv, index

        cvv, index = no_graph(x, cvv, index)
        res = self.lin_final(res)

        return res, x[:, -1:, :], index, cvv

    def forward(self, x: torch.Tensor, state: torch.Tensor, index: int, state_cvv: torch.Tensor):
        x = self.lin_in(x)
        xE: torch.Tensor = x - state
        xE: torch.Tensor = self.lin_dif(xE)
        cvv = state_cvv + xE
        if index % 32 == 0:
            cvv = self.norm(cvv)
            cvv = self.lin_intermediate(cvv)
        index = index + 1
        cyv: torch.Tensor = self.lin_final(cvv)

        return cyv, x, index, cvv
