import random
import torch
from torch import nn
from .attention import Attention
from .feedforward import PreNet


class Decoder(nn.Module):
    def __init__(self, num_mels=80, prenet_dim=256, embed_dim=512,
                 attention_lstm_dim=1024, decoder_lstm_dim=1024,
                 attention_dim=128, attention_temp=0.08, attention_dropout=0.1,
                 dropout=0.5, max_frames=870, threshold=0.5, frames_per_char=5.75):
        super(Decoder, self).__init__()

        self.num_mels = num_mels
        self.prenet_dim = prenet_dim
        self.embed_dim = embed_dim
        self.attention_dim = attention_dim
        self.attention_lstm_dim = attention_lstm_dim
        self.decoder_lstm_dim = decoder_lstm_dim
        self.max_frames = max_frames
        self.threshold = threshold
        self.frames_per_char = frames_per_char
        self.teacher_forcing = None

        self.prenet = PreNet(dims=[num_mels, prenet_dim, prenet_dim], dropout=dropout)
        self.attention_lstm = nn.LSTMCell(input_size=prenet_dim + embed_dim,
                                          hidden_size=attention_lstm_dim)

        self.attention = Attention(embed_dim, attention_dim, attention_lstm_dim,
                                   attention_temp, attention_dropout)
        self.decoder_lstm = nn.LSTMCell(input_size=attention_lstm_dim + embed_dim,
                                        hidden_size=decoder_lstm_dim)

        self.spec_fc = nn.Linear(in_features=decoder_lstm_dim + embed_dim,
                                 out_features=num_mels)
        self.stop_fc = nn.Linear(in_features=decoder_lstm_dim + embed_dim,
                                 out_features=1)

    def init_states(self, batch_size, device):
        decoder_outputs = torch.zeros((batch_size, self.num_mels)).to(device)
        attention_context = torch.zeros((batch_size, self.embed_dim)).to(device)

        attention_hidden = torch.zeros((batch_size, self.attention_lstm_dim)).to(device)
        attention_cell = torch.zeros((batch_size, self.attention_lstm_dim)).to(device)

        decoder_hidden = torch.zeros((batch_size, self.decoder_lstm_dim)).to(device)
        decoder_cell = torch.zeros((batch_size, self.decoder_lstm_dim)).to(device)

        return decoder_outputs, attention_context, attention_hidden, attention_cell, \
               decoder_hidden, decoder_cell

    def forward(self, encoder_outputs, lengths, melspecs):
        # encoder_outputs: (batch_size, char_length, embed_dim)
        # lengths: (batch_size, )
        # melspecs: (batch_size, frames_length, num_mels)

        batch_size, char_length, _ = encoder_outputs.shape
        frames_length = melspecs.shape[1]
        device = encoder_outputs.device

        # initialize all states with zeros
        decoder_outputs, attention_context, attention_hidden, attention_cell, \
        decoder_hidden, decoder_cell = self.init_states(batch_size, device)

        # prepare K, V and mask for attention
        K = self.attention.WK(encoder_outputs)
        V = self.attention.WV(encoder_outputs)
        # K: (batch_size, char_length, attention_dim)
        # V: (batch_size, char_length, embed_dim)

        mask = torch.arange(char_length).view(1, char_length) >= lengths.view(batch_size, 1)
        mask = mask.unsqueeze(1).to(device)
        # mask: (batch_size, 1, char_length)

        output_melspecs, output_probs = [], []
        for i in range(frames_length):
            # teacher forcing
            if i > 0 and random.random() < self.teacher_forcing:
                decoder_outputs = melspecs[:, i - 1]

            # PreNet for previous step
            prenet_outputs = self.prenet(decoder_outputs)
            # prenet_outputs: (batch_size, prenet_dim)

            # attention LSTM
            attention_lstm_inputs = torch.cat([prenet_outputs, attention_context], dim=1)
            # attention_lstm_inputs: (batch_size, prenet_dim + embed_dim)

            attention_hidden, attention_cell = self.attention_lstm(attention_lstm_inputs,
                                                                   (attention_hidden, attention_cell))
            # attention_hidden, attention_cell: (batch_size, attention_lstm_dim)

            step = (i + 1) / (char_length * self.frames_per_char)
            attention_context = self.attention(query=attention_hidden.unsqueeze(1),
                                               K=K, V=V, step=step, mask=mask)
            attention_context = attention_context.squeeze(1)
            # attention_context: (batch_size, embed_dim)

            decoder_lstm_inputs = torch.cat([attention_hidden, attention_context], dim=1)
            # decoder_lstm_inputs: (batch_size, attention_lstm_dim + embed_dim)

            decoder_hidden, decoder_context = self.decoder_lstm(decoder_lstm_inputs,
                                                                (decoder_hidden, decoder_cell))
            # decoder_hidden, decoder_cell: (batch_size, decoder_lstm_dim)

            frame_features = torch.cat([decoder_hidden, attention_context], dim=1)
            # frame_features: (batch_size, decoder_lstm_dim + embed_dim)

            decoder_outputs = self.spec_fc(frame_features)
            stop_probs = torch.sigmoid(self.stop_fc(frame_features))
            # decoder_outputs: (batch_size, num_mels)
            # stop_probs: (batch_size, 1)

            output_melspecs += [decoder_outputs.unsqueeze(1)]
            output_probs += [stop_probs]

        output_melspecs = torch.cat(output_melspecs, dim=1)
        output_probs = torch.cat(output_probs, dim=1)
        # output_melspecs: (batch_size, frames_length, prenet_dim)
        # output_probs: (batch_size, frames_length)

        return output_melspecs, output_probs

    def inference(self, encoder_outputs, lengths):
        # encoder_outputs: (batch_size, char_length, embed_dim)
        # lengths: (batch_size, )

        batch_size, char_length, _ = encoder_outputs.shape
        device = encoder_outputs.device

        # initialize all states with zeros
        decoder_outputs, attention_context, attention_hidden, attention_cell, \
        decoder_hidden, decoder_cell = self.init_states(batch_size, device)

        # prepare K, V and mask for attention
        K = self.attention.WK(encoder_outputs)
        V = self.attention.WV(encoder_outputs)
        # K: (batch_size, char_length, attention_dim)
        # V: (batch_size, char_length, embed_dim)

        mask = torch.arange(char_length).view(1, char_length) >= lengths.view(batch_size, 1)
        mask = mask.unsqueeze(1).to(device)
        # mask: (batch_size, 1, char_length)

        output_melspecs, output_probs = [], []
        for i in range(self.max_frames):
            # PreNet for previous step
            prenet_outputs = self.prenet(decoder_outputs)
            # prenet_outputs: (batch_size, prenet_dim)

            # attention LSTM
            attention_lstm_inputs = torch.cat([prenet_outputs, attention_context], dim=1)
            # attention_lstm_inputs: (batch_size, prenet_dim + embed_dim)

            attention_hidden, attention_cell = self.attention_lstm(attention_lstm_inputs,
                                                                   (attention_hidden, attention_cell))
            # attention_hidden, attention_cell: (batch_size, attention_lstm_dim)

            step = (i + 1) / (char_length * self.frames_per_char)
            attention_context = self.attention(query=attention_hidden.unsqueeze(1),
                                               K=K, V=V, step=step, mask=mask)
            attention_context = attention_context.squeeze(1)
            # attention_context: (batch_size, embed_dim)

            decoder_lstm_inputs = torch.cat([attention_hidden, attention_context], dim=1)
            # decoder_lstm_inputs: (batch_size, attention_lstm_dim + embed_dim)

            decoder_hidden, decoder_context = self.decoder_lstm(decoder_lstm_inputs,
                                                                (decoder_hidden, decoder_cell))
            # decoder_hidden, decoder_cell: (batch_size, decoder_lstm_dim)

            frame_features = torch.cat([decoder_hidden, attention_context], dim=1)
            # frame_features: (batch_size, decoder_lstm_dim + embed_dim)

            decoder_outputs = self.spec_fc(frame_features)
            stop_probs = torch.sigmoid(self.stop_fc(frame_features))
            # spec_frames: (batch_size, num_mels)
            # stop_probs: (batch_size, 1)

            output_melspecs += [decoder_outputs.unsqueeze(1)]
            output_probs += [stop_probs]

            if i > 0 and torch.all(stop_probs > self.threshold):
                break

        output_melspecs = torch.cat(output_melspecs, dim=1)
        output_probs = torch.cat(output_probs, dim=1)
        # output_melspecs: (batch_size, frames_length, prenet_dim)
        # output_probs: (batch_size, frames_length)

        return output_melspecs, output_probs

