import torch
from torch import nn
from .mlp import MLP
from .pos_embedding import PositionEmbedding


class STELLA(nn.Module):

    r"""STELLA architecture."""

    def __init__(self, **model_args):
        super().__init__()

        # attributes
        self.num_nodes = model_args["num_nodes"]  # number of stations
        self.num_features = model_args["num_features"] # number of variables
        self.input_len = model_args["input_len"] # input sequence length
        self.d_model = model_args["d_model"] # hidden dimension
        self.output_len = model_args["output_len"] # output sequence length
        self.num_layer = model_args["num_layer"] # number of layers
        self.if_rel = model_args["if_rel"]  # whether to use relative positional encoding, True or False
        self.res_conn = model_args["res_conn"] # whether to add residual connection to MLP, True or False
        self.root_path = model_args["root_path"] # root path of the positional information
        self.dropout = model_args["dropout"] # dropout rate

        # input embedding layer
        self.input_embedding = nn.Linear(self.input_len, self.d_model)

        # positional encoding
        self.pos_embedding = PositionEmbedding(self.d_model, self.num_nodes, self.root_path, self.if_rel)

        # MLP
        self.encoder = nn.Sequential(
            *[MLP(self.d_model, self.d_model, self.res_conn, self.dropout) for _ in range(self.num_layer)]
        )

        # regression
        self.output_layer = nn.Linear(self.d_model, self.output_len)

    def forward(self, history_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """forward

                :param x: history data with shape [B, L, N, C]:

                Returns:
                    torch.Tensor: prediction with shape [B, F, N, C]

                """

        x = history_data[..., 0: self.num_features]
        x_mark = history_data[:, 0, 0, self.num_features:]

        # embedding
        x = x.transpose(1, -1)  # [B, C, N, L]
        x = self.input_embedding(x)
        x = self.pos_embedding(x, x_mark)  # [B, C, N, D]

        # encoding
        output = self.encoder(x)  # [B, C, N, D]

        # output
        prediction = self.output_layer(output).transpose(1, -1)  # [B, F, N, C]

        return prediction