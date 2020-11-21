from torch import nn
from .encoder import Encoder
from .decoder import Decoder
from .feedforward import PostNet


class Tacotron(nn.Module):
    def __init__(self, num_chars, num_mels=80, embed_dim=512, prenet_dim=256,
                 attention_lstm_dim=1024, decoder_lstm_dim=1024, attention_dim=128, attention_temp=0.08,
                 encoder_layers=3, kernel_size=5, postnet_layers=3, postnet_channels=512,
                 attention_dropout=0.1, dropout=0.5, max_frames=870, threshold=0.5):
        super(Tacotron, self).__init__()

        self.encoder = Encoder(num_chars, embed_dim, encoder_layers,
                               kernel_size, dropout)

        self.decoder = Decoder(num_mels, prenet_dim, embed_dim,
                               attention_lstm_dim, decoder_lstm_dim,
                               attention_dim, attention_temp, attention_dropout,
                               dropout, max_frames, threshold)

        self.postnet = PostNet(postnet_layers, num_mels, postnet_channels,
                               kernel_size, dropout)

    def forward(self, chars, lengths, melspecs):
        # chars: (batch_size, char_length)
        # lengths: (batch_size, )
        # melspecs: (batch_size, frames_length, num_mels)

        encoder_outputs = self.encoder(chars, lengths)
        # encoder_outputs: (batch_size, char_length, embed_dim)

        decoder_melspecs, decoder_probs = self.decoder(chars, lengths, melspecs)
        # decoder_melspecs: (batch_size, frames_length, num_mels)
        # decoder_probs: (batch_size, frames_length)

        postnet_melspecs = self.postnet(decoder_melspecs)
        # postnet_melspecs: (batch_size, frames_length, num_mels)

        return decoder_melspecs, postnet_melspecs, decoder_probs

    def inference(self, char, lengths):
        # chars: (batch_size, char_length)
        # lengths: (batch_size, )

        encoder_outputs = self.encoder(chars, lengths)
        # encoder_outputs: (batch_size, char_length, embed_dim)

        decoder_melspecs, decoder_probs = self.decoder.inference(chars, lengths)
        # decoder_melspecs: (batch_size, frames_length, num_mels)
        # decoder_probs: (batch_size, frames_length)

        postnet_melspecs = self.postnet(decoder_melspecs)
        # postnet_melspecs: (batch_size, frames_length, num_mels)

        return postnet_melspecs, decoder_probs


def tacotron(num_chars, params):
    return Tacotron(num_chars, params.num_mels, params.embed_dim, params.prenet_dim,
                 params.attention_lstm_dim, params.decoder_lstm_dim, params.attention_dim, params.attention_temp,
                 params.encoder_layers, params.kernel_size, params.postnet_layers, params.postnet_channels,
                 params.attention_dropout, params.dropout, params.max_frames, params.threshold)

