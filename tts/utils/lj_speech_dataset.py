import torch
import torchaudio


class LJSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, labels, alphabet, params):
        super(LJSpeechDataset, self).__init__()

        self.alphabet = alphabet
        self.labels = labels.sort_values(by='transcription', ignore_index=True,
                                         key=lambda column: column.str.len())

        self.data_root = params.data_root
        self.max_audio_length = max_audio_length
        self.max_chars_length = max_chars_length
        self.sample_rate = sample_rate

    def pad_sequence(self, sequence, max_length, fill=0.0, dtype=torch.float):
        padded_sequence = torch.full((max_length, ), fill_value=fill, dtype=dtype)
        sequence_length = min(sequence.shape[0], max_length)
        padded_sequence[:sequence_length] = sequence[:sequence_length]
        return padded_sequence

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        audio_info = self.labels.loc[index]
        waveform, sample_rate = torchaudio.load(self.root + audio_info.path + '.wav')

        if sample_rate != self.sample_rate:
            raise ValueError('Wrong sample rate!')

        waveform = waveform.view(-1)
        audio_length = min(waveform.shape[0], self.max_audio_length)
        waveform = self.pad_sequence(waveform, self.max_audio_length)

        chars = self.alphabet.string_to_indices(audio_info.transcription)
        chars_length = chars.shape[0]
        chars = self.pad_sequence(chars, self.max_chars_length, dtype=torch.long)

        audio_length = torch.tensor(audio_length, dtype=torch.long)
        chars_length = torch.tensor(chars_length, dtype=torch.long)
        return waveform, chars, audio_length, chars_length

