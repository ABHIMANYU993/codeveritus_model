import torch
from transformers import RobertaTokenizer
import torch.nn as nn
from model import Autoencoder

# Assuming you already have the Autoencoder class definition from previous code

# Define the path to the saved model
model_path = "C:/Users/ABHIMANYU/PycharmProjects/cypherhackathon/Saved_model/otherapproach.pth"

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# Load the trained model
autoencoder = Autoencoder(device)
autoencoder.load_state_dict(torch.load(model_path, weights_only=True))
autoencoder.eval()  # Set the model to evaluation mode


# Function to detect anomalies in new code samples
def detect_anomalies(code_sample):
    # Tokenize the new code sample
    tokenized = tokenizer.encode_plus(
        code_sample,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )['input_ids'].to(device)

    # Ensure tokenized input is float
    tokenized = tokenized.float()

    # Pass the tokenized input through the trained autoencoder
    with torch.no_grad():  # No need to track gradients during inference
        reconstructed = autoencoder(tokenized)
        loss_fn = nn.MSELoss()
        loss = loss_fn(reconstructed, tokenized)

    # Return the reconstruction error as the anomaly score
    return loss.item()


# Example usage with a new code sample
new_code_sample = '''
#include <stdio.h>

int findClosestNegative(int* arr, int size) {
    int left = 0;
    int right = size - 1;
    int closestNegative = arr[left]; // Initialize with the first element

    while (left <= right) {
        int mid = left + (right - left) / 2;

        // Check if the mid value is negative
        if (arr[mid] < 0) {
            // If found a negative number, update closestNegative if it's closer to zero
            if (arr[mid] > closestNegative) { // since arr is ascending
                closestNegative = arr[mid];
            }
            // Move right to find potentially closer negative numbers
            left = mid + 1;
        } else {
            // Move left to find negative numbers