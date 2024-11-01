import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, device):
        super(Autoencoder, self).__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64)
        ).to(self.device)

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        ).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# clean up exception handling block

# improve error message formatting

# update batch normalization momentum