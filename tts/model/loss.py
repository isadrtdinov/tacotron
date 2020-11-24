import torch
from torch import nn


class MaskedLoss(nn.Module):
    def __init__(self, loss):
        super(MaskedLoss, self).__init__()
        self.loss = loss(reduction='none')

    def forward(self, predicts, targets, mask):
        weights = torch.ones_like(predicts)
        weights = weights.masked_fill(mask, 0.0)
        norm = weights.sum().item()
        return torch.sum(weights * self.loss(predicts, targets)) / norm


class TacotronLoss(nn.Module):
    def __init__(self):
        super(TacotronLoss, self).__init__()
        self.mse = MaskedLoss(nn.MSELoss)
        self.bce = MaskedLoss(nn.BCELoss)

    def forward(self, predicts, targets, mask):
        decoder_melspecs, postnet_melspecs, decoder_probs, _ = predicts
        melspecs, stop_labels = targets

        decoder_mse = self.mse(decoder_melspecs, melspecs, mask.unsqueeze(2))
        postnet_mse = self.mse(postnet_melspecs, melspecs, mask.unsqueeze(2))
        probs_bce = self.bce(decoder_probs, stop_labels, mask)
        loss = decoder_mse + postnet_mse + probs_bce

        components = (decoder_mse.item(), postnet_mse.item(),
                      probs_bce.item(), loss.item())
        return loss, components

