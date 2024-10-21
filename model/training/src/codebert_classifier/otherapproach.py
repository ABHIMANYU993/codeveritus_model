import os
from transformers import RobertaTokenizer
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.optim as optim


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
        x = x.to(self.device)  # Ensure the input is on the right device
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Check if CUDA is available
print("CUDA available: ", torch.cuda.is_available())

# If CUDA is available, print the GPU device name
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")


# Function to load and tokenize the code files
def load_and_tokenize_c_files(folder_path):
    tokens = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.c'):
            file_path = os.path.join(folder_path, filename)