import torch


class Alphabet(object):
    def __init__(self, tokens=''):
        self.index_to_token = {}
        self.index_to_token.update({i: tokens[i] for i in range(len(tokens))})
        self.token_to_index = {token: index
                               for index, token in self.index_to_token.items()}

    def string_to_indices(self, string):
        return torch.tensor([self.token_to_index[token] for token in string \
                             if token in self.token_to_index], dtype=torch.long)

    def indices_to_string(self, indices):
        return ''.join(self.index_to_token[index.item()] for index in indices)

