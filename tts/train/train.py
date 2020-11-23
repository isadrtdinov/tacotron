import random
import torch
import wandb
from ..model.loss import TacotronLoss
from ..utils.vocoder import Vocoder
from ..utils.spectrogram import MelSpectrogram, get_spectrogram_lengths


def process_batch(model, optimizer, criterion, batch, params, train=True):
    chars, char_lengths, melspecs, melspec_lengths = batch

    batch_size, frames_length, _ = melspecs.shape
    mask = torch.arange(frames_length).view(1, frames_length) >= \
           melspec_lengths.view(batch_size, 1)
    mask = mask.to(params.device)

    stop_labels = torch.arange(1, frames_length + 1).view(1, frames_length) / \
                  melspec_lengths.view(batch_size, 1) - 1.0
    stop_labels = torch.exp(-stop_labels ** 2 / params.labels_temp).to(params.device)

    optimizer.zero_grad()
    with torch.set_grad_enabled(train):
        predicts = model(chars, char_lengths, melspecs)
        targets = (melspecs, stop_labels)
        loss, components = criterion(predicts, targets, mask)

        if train:
            loss.backward()
            optimizer.step()

    return components


def process_epoch(model, optimizer, criterion, spectrogramer, loader, params, train=True):
    model.train() if train else model.eval()

    running_loss = [0.0] * 4

    for raw_batch in loader:
        waveforms, chars, audio_lengths, char_lengths = raw_batch

        waveforms = waveforms[:, :audio_lengths.max()].to(params.device)
        chars = chars[:, :char_lengths.max()].to(params.device)

        melspecs = spectrogramer(waveforms).transpose(1, 2)
        melspec_lengths = get_spectrogram_lengths(audio_lengths, params)

        batch = (chars, char_lengths, melspecs, melspec_lengths)
        components = process_batch(model, optimizer, criterion, batch, params, train)

        running_loss = [cumulative_loss + loss * waveforms.shape[0]
                        for cumulative_loss, loss in zip(running_loss, components)]

    running_loss = [cumulative_loss / len(loader.dataset)
                    for cumulative_loss in running_loss]

    return running_loss


def generate_example(model, spectrogramer, loader, vocoder, params):
    model.eval()

    rand_index = random.randrange(len(loader.dataset))
    waveform, chars, audio_length, char_length = loader.dataset[rand_index]

    waveform = waveform[:audio_length.item()].unsqueeze(0).to(params.device)
    chars = chars[:char_length.item()].unsqueeze(0).to(params.device)
    char_length = char_length.unsqueeze(0)

    with torch.no_grad():
        target_melspec = spectrogramer(waveform)
        _, predicted_melspec, predicted_probs = model(chars, char_length, target_melspec.transpose(1, 2))
        predicted_melspec = predicted_melspec.transpose(1, 2)
        predicted_waveform = vocoder.inference(predicted_melspec)

    target_melspec = target_melspec.squeeze(0).cpu().numpy()
    predicted_melspec = predicted_melspec.squeeze(0).cpu().numpy()
    predicted_waveform = predicted_waveform.squeeze(0).cpu().numpy()
    predicted_probs = predicted_probs.squeeze(0).cpu().numpy()

    text = loader.dataset.alphabet.indices_to_string(chars.squeeze(0).cpu())
    probs_plot = [[frame, prob] for frame, prob in enumerate(predicted_probs, 1)]
    probs_plot = wandb.Table(data=probs_plot, columns=['frame', 'prob'])

    example = {'ground truth spectrogram': wandb.Image(target_melspec),
               'predicted spectrogram': wandb.Image(predicted_melspec),
               'predicted audio': wandb.Audio(predicted_waveform, sample_rate=params.sample_rate),
               'ground truth text': wandb.Table(data=[[text]], columns=['text']),
               'stop probability': wandb.plot.line(probs_plot, 'frame', 'prob')}

    return example


def train(model, optimizer, train_loader, valid_loader, params):
    criterion = TacotronLoss()
    vocoder = Vocoder(params.vocoder_file)
    spectrogramer = MelSpectrogram(params).to(params.device)

    for epoch in range(params.start_epoch, params.start_epoch + params.num_epochs):
        #train_loss = process_epoch(model, optimizer, criterion, spectrogramer,
        #                           train_loader, params, train=True)

        #valid_loss = process_epoch(model, optimizer, criterion, spectrogramer,
        #                           valid_loader, params, train=False)
        train_loss, valid_loss = [0] * 4, [0] * 4
        if params.use_wandb:
            vocoder = vocoder.to(params.device)
            example = generate_example(model, spectrogramer, valid_loader, vocoder, params)
            vocoder = vocoder.cpu()

            example.update({'train decoder mse': train_loss[0], 'train postnet mse': train_loss[1],
                            'train probs bce': train_loss[2], 'train total loss': train_loss[3],
                            'valid decoder mse': valid_loss[0], 'valid postnet mse': valid_loss[1],
                            'valid probs bce': valid_loss[2], 'valid total loss': valid_loss[3]})

            wandb.log(example)

        torch.save({
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
        }, params.checkpoint_template.format(epoch))

        return
