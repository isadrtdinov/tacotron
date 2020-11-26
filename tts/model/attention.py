import math
import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, embed_dim=512, attention_dim=128,
                 attention_lstm_dim=1024, dropout=0.1):
        super(Attention, self).__init__()

        self.WQ = nn.Linear(attention_lstm_dim, attention_dim, bias=False)
        self.WK = nn.Linear(embed_dim, attention_dim, bias=False)
        self.WV = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, K, V, mask=None):
        # query: (batch_size, 1, attention_lstm_dim)
        # K = WK(key): (batch_size, char_length, attention_dim)
        # V = WV(value): (batch_size, char_length, embed_dim)
        # mask: (batch_size, 1, char_length)

        Q = self.WQ(query)
        # Q: (batch_size, 1, attention_dim)

        norm_factor = math.sqrt(Q.shape[-1])
        attention_score = torch.bmm(Q, K.transpose(1, 2)) / norm_factor
        # attention_score: (batch_size, 1, char_length)

        if mask is not None:
            attention_score = attention_score.masked_fill(mask, -math.inf)

        attention_probs = nn.functional.softmax(attention_score, dim=-1)
        attention_probs = self.dropout(attention_probs)
        soft_argmax = torch.bmm(attention_probs, V)
        # attention_probs: (batch_size, 1, char_length)
        # soft_argmax: (batch_size, length, attention_dim)

        return soft_argmax, attention_probs

