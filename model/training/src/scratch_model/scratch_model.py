import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

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

    @classmethod
    def load_pretrained(cls, tokenizer_path):
        with open(os.path.join(tokenizer_path, 'vocab.txt'), 'r') as f:
            vocab = f.read().splitlines()
        tokenizer = cls(vocab_size=len(vocab))
        tokenizer.token_to_id = {token: i for i, token in enumerate(vocab)}
        return tokenizer


# Step 2: Positional Encoding (Replicating Roberta Positional Encoding)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(pos * div_term)
        self.encoding[:, 1::2] = torch.cos(pos * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :].to(x.device)


# Step 3: Transformer Block (Replicating Roberta Transformer Blocks)
class TransformerBlock(nn.Module):
    def __init__(self, d_model=768, num_heads=12, ff_hidden_dim=3072):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.GELU(),
            nn.Linear(ff_hidden_dim, d_model)
        )

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x


# Step 4: CodeBERT Classifier (Replicating RobertaForSequenceClassification)