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
    def __init__(self, guide_temp, attention_coef):
        super(TacotronLoss, self).__init__()
        self.mse = MaskedLoss(nn.MSELoss)
        self.bce = MaskedLoss(nn.BCELoss)
        self.guide_temp = guide_temp
        self.attention_coef = attention_coef

    def forward(self, predicts, targets, mask):
        decoder_melspecs, postnet_melspecs, decoder_probs, attention = predicts
        melspecs, stop_labels = targets

        decoder_mse = self.mse(decoder_melspecs, melspecs, mask.unsqueeze(2))
        postnet_mse = self.mse(postnet_melspecs, melspecs, mask.unsqueeze(2))
        probs_bce = self.bce(decoder_probs, stop_labels, mask)

        _, T, N = attention.shape
        frames_pos = (torch.arange(1, T + 1) / T).view(1, T, 1)
        char_pos = (torch.arange(1, N + 1) / N).view(1, 1, N)

        guide_mask = 1.0 - torch.exp(-(frames_pos - char_pos) ** 2 / self.guide_temp)
        attention = attention * guide_mask.to(attention.device)
        attention_loss = self.attention_coef * attention.mean()

        loss = decoder_mse + postnet_mse + probs_bce + attention_loss
        components = (decoder_mse.item(), postnet_mse.item(), probs_bce.item(),
                      attention_loss.item(), loss.item())

        return loss, components

