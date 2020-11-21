from torch import nn


class Conv1DLayer(nn.Module):
    def __init__(self, in_channels=512, out_channels=512, kernel_size=5,
                 activation='relu', dropout=0.5):
        super(Conv1DLayer, self).__init__()

        padding = (kernel_size - 1) // 2 # padding 'same'
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, padding=padding)
        self.batch_norm = nn.BatchNorm1d(num_features=out_channels)
        self.dropout = nn.Dropout(dropout)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = nn.Identity()
        else:
            raise ValueError('Unknown activation!')

    def forward(self, inputs):
        # inputs: (batch_size, in_channels, max_length)

        outputs = self.batch_norm(self.conv(inputs))
        outputs = self.dropout(self.activation(outputs))
        # outputs: (batch_size, out_channels, max_length)

        return outputs

