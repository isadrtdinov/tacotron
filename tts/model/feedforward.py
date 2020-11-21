from torch import nn
import torch.nn.functional as F
from .layers import Conv1DLayer


class PreNet(nn.Module):
    def __init__(self, dims=[80, 256, 256], dropout=0.5):
        super(PreNet, self).__init__()

        self.layers = nn.ModuleList([
            nn.Linear(dims[i - 1], dims[i]) for i in range(1, len(dims))
        ])
        self.dropout = dropout

    def forward(self, inputs):
        # inputs: (batch_size, *, num_mels)

        outputs = inputs
        for layer in self.layers:
            outputs = F.dropout(F.relu(layer(outputs)), self.dropout, training=True)
        # outputs: (batch_size, *, prenet_dim)

        return outputs


class PostNet(nn.Module):
    def __init__(self, conv_layers=5, num_mels=80, num_channels=512,
                 kernel_size=5, dropout=0.5):
        assert conv_layers >= 2
        super(PostNet, self).__init__()

        self.layers = [Conv1DLayer(in_channels=num_mels, out_channels=num_channels,
                                   kernel_size=kernel_size, activation='tanh', dropout=dropout)]
        self.layers += [Conv1DLayer(in_channels=num_channels, out_channels=num_channels,
                                    kernel_size=kernel_size, activation='tanh', dropout=dropout)
                        for _ in range(1, conv_layers - 1)]

        self.layers += [Conv1DLayer(in_channels=num_channels, out_channels=num_mels,
                                    kernel_size=kernel_size, activation='none', dropout=0.0)]

        self.layers = nn.ModuleList(self.layers)

    def forward(self, inputs):
        # inputs: (batch_size, num_frames, num_mels)

        outputs = inputs.transpose(1, 2)
        for layer in self.layers:
            outputs = layer(outputs)
        outputs = outputs.transpose(1, 2)
        # outputs: (batch_size, num_frames, num_mels)

        return outputs

