import math
import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, embed_dim=512, attention_dim=128,
                 attention_lstm_dim=1024, temp=0.08, dropout=0.1):
        super(Attention, self).__init__()
        self.temp = temp

        self.WQ = nn.Linear(attention_lstm_dim, attention_dim, bias=False)
        self.WK = nn.Linear(embed_dim, attention_dim, bias=False)
        self.WV = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, K, V, step=None, mask=None):
        # query: (batch_size, 1, attention_lstm_dim)
        # K = WK(key): (batch_size, char_length, attention_dim)
        # V = WV(value): (batch_size, char_length, embed_dim)
        # mask: (batch_size, 1, char_length)

        Q = self.WQ(query)
        # Q: (batch_size, 1, attention_dim)

        norm_factor = math.sqrt(Q.shape[-1])
        attention = torch.bmm(Q, K.transpose(1, 2)) / norm_factor
        # attention: (batch_size, 1, char_length)

        if step is not None:
            N = attention.shape[-1]
            frames_pos = torch.tensor([step]).view(1, 1)
            chars_pos = (torch.arange(1, N + 1) / N).view(1, N)

            guide_mask = torch.exp(-(frames_pos - chars_pos) ** 2 / self.temp)
            attention = attention * guide_mask.to(attention.device)

        if mask is not None:
            attention = attention.masked_fill(mask, -math.inf)

        attention_score = nn.functional.softmax(attention, dim=-1)
        attention_score = self.dropout(attention_score)
        soft_argmax = torch.bmm(attention_score, V)
        # soft_argmax: (batch_size, length, attention_dim)

        return soft_argmax

