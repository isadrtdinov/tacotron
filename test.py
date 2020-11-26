import sys
import string
import warnings
import torch
import torchaudio
import soundfile as sf
from tts.model import tacotron
from tts.test import test
from tts.utils import Alphabet
from config import set_params


def main():
    # set params
    params = set_params()
    sys.path.append(params.vocoder_dir)
    warnings.filterwarnings('ignore')
    params.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # tokenize text
    alphabet = Alphabet(tokens=string.ascii_lowercase + ' !\"\'(),-.:;?[]')
    with open(params.example_text) as file_object:
        text = file_object.readline().lower()[:-1]
        text = alphabet.string_to_indices(text)

    # load model
    model = tacotron(len(alphabet.index_to_token), params).to(params.device)
    checkpoint = torch.load(params.model_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])

    # generate and save audio
    waveform = test(model, text, params)
    sf.write(params.example_audio, waveform, params.sample_rate)


if __name__ == '__main__':
    main()

