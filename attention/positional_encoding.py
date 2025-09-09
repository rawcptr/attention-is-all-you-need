import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(
        self, d_model: int, seq_len: int, dropout: float, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        # offset vectors
        pe = torch.zeros((seq_len, d_model))
        position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.tensor(10_000.0).log().div(d_model))
        )

        inner = position * div_term
        # even -> sin, odd -> cos
        pe[:, 0::2] = torch.sin(inner)
        pe[:, 1::2] = torch.cos(inner)

        pe = pe.unsqueeze(0)
        self.pe = pe
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
