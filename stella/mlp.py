import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, res_conn, dropout):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.res_conn = res_conn

        if self.res_conn:
            self.model = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Dropout(p=dropout)
            )

        else:
            self.model = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        if self.res_conn:
            out = self.model(x) + x
        else:
            out = self.model(x)
        return out
