import torch
from ..utils.vocoder import Vocoder


def test(model, text, params):
    vocoder = Vocoder(params.vocoder_file).to(params.device)
    text = text.unsqueeze(0).to(params.device)
    length = torch.tensor([text.shape[1]])

    melspec, _, _ = model.inference(text, length)
    waveform = vocoder.inference(melspec.transpose(1, 2))
    return waveform.squeeze(0).clamp(-1.0, 1.0).cpu().numpy()

