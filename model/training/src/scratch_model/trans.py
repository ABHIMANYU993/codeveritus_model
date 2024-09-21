import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Step 1: Custom Tokenizer (Replicating RobertaTokenizer)
class RobertaTokenizerFromScratch:
    def __init__(self, vocab_size=30522, max_length=512):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.token_to_id = {chr(i + 97): i + 1 for i in range(26)}
        self.token_to_id.update({
            '[PAD]': 0,
            '[UNK]': vocab_size - 1,
            '[CLS]': vocab_size - 2,
            '[SEP]': vocab_size - 3,
            '[MASK]': vocab_size - 4
        })

    def encode(self, text):
        text = text.lower()
        tokens = [self.token_to_id.get(char, self.token_to_id['[UNK]']) for char in text]
        tokens = [self.token_to_id['[CLS]']] + tokens + [self.token_to_id['[SEP]']]
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens += [self.token_to_id['[PAD]']] * (self.max_length - len(tokens))
        return tokens

    def decode(self, token_ids):
        id_to_token = {v: k for k, v in self.token_to_id.items()}
        return ''.join([id_to_token.get(token_id, '?') for token_id in token_ids if token_id in id_to_token])


# Step 2: Positional Encoding (Replicating Roberta Positional Encoding)