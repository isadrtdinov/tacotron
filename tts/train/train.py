import torch
import wandb
from ..model.loss import TacotronLoss
from ..utils.vocoder import Vocoder
from ..utils.spectrogram import MelSpectrogram, get_spectrogram_lengths


def process_batch(model, optimizer, criterion, batch, train=True):
    chars, char_lengths, melspecs, melspec_lengths = batch

    batch_size, frames_length, _ = melspecs.shape
    mask = torch.arange(frames_length).view(1, frames_length) >= \
           melspec_lengths.view(batch_size, 1)
    mask = mask.to(chars.device)

    stop_labels = torch.arange(frames_length).view(1, frames_length) >= \
                  (melspec_lengths - 1).view(batch_size, 1)
    stop_labels = stop_labels.to(torch.float).to(chars.device)

    optimizer.zero_grad()
    with torch.set_grad_enabled(train):
        predicts = model(chars, char_lengths, melspecs)
        targets = (melspecs, stop_labels)
        loss = criterion(predicts, targets, mask)

        if train:
            loss.backward()
            optimizer.step()

    return loss.item()


def process_epoch(model, optimizer, criterion, spectrogramer, loader, params, train=True):
    model.train() if train else model.eval()

    running_loss = 0.0

    for raw_batch in loader:
        waveforms, chars, audio_lengths, char_lengths = raw_batch

        waveforms = waveforms[:, :audio_lengths.max()].to(params.device)
        chars = chars[:, :char_lengths.max()].to(params.device)

        melspecs = spectrogramer(waveforms).transpose(1, 2)
        melspec_lengths = get_spectrogram_lengths(audio_lengths, params)

        batch = (chars, char_lengths, melspecs, melspec_lengths)
        loss = process_batch(model, optimizer, criterion, batch, train)

        running_loss += loss * waveforms.shape[0]

    running_loss /= len(loader.dataset)
    return running_loss


def train(model, optimizer, train_loader, valid_loader, params):
    criterion = TacotronLoss()
    vocoder = Vocoder(params.vocoder_file)
    spectrogramer = MelSpectrogram(params).to(params.device)

    for epoch in range(params.start_epoch, params.start_epoch + params.num_epochs):
        train_loss = process_epoch(model, optimizer, criterion, spectrogramer,
                                   train_loader, params, train=True)

        valid_loss = process_epoch(model, optimizer, criterion, spectrogramer,
                                   valid_loader, params, train=False)

