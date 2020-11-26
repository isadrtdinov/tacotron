from ..utils.vocoder import Vocoder


def test(model, text, params):
    vocoder = Vocoder(params.vocoder_file).to(params.device)
    text = text.unsqueeze(0).to(device)
    length = torch.tensor([text.shape[1]])

    melspec, _, _ = model.inference(text, length)
    waveform = vocoder.inference(melspec).unsqueeze(0).cpu()
    return waveform

