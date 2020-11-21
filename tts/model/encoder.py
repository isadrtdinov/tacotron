from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .layers import Conv1DLayer


class Encoder(nn.Module):
    def __init__(self, num_chars, embed_dim=512,
                 conv_layers=3, kernel_size=5, dropout=0.5):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_chars, embed_dim)
        self.conv_layers = nn.ModuleList([
            Conv1DLayer(embed_dim, embed_dim, kernel_size, 'relu', dropout)
            for _ in range(conv_layers)
        ])

        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=embed_dim // 2,
                            batch_first=True, bidirectional=True)

    def forward(self, chars, lengths):
        # chars: (batch_size, char_length)
        # lengths: (batch_size, )

        embeds = self.embedding(chars)
        embeds = embeds.transpose(1, 2)
        # embeds: (batch_size, embed_dim, char_length)

        features = embeds
        for layer in self.conv_layers:
            features = layer(features)
        features = features.transpose(1, 2)
        # features: (batch_size, char_length, embed_dim)

        features = pack_padded_sequence(features, lengths, batch_first=True,
                                        enforce_sorted=False)
        features, _ = self.lstm(features)
        features, _ = pad_packed_sequence(features, batch_first=True)
        # features: (batch_size, char_length, embed_dim)

        return features

