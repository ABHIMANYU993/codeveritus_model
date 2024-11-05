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
            # Open the file with UTF-8 encoding and error handling
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                code = file.read()
                # Tokenize the code
                encoded = tokenizer.encode_plus(
                    code,
                    add_special_tokens=True,
                    max_length=512,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                tokens.append(encoded['input_ids'])
    return tokens


# Example usage for AI-generated code
tokens = load_and_tokenize_c_files("C:/Users/ABHIMANYU/PycharmProjects/cypherhackathon/code_sample_public/3_GPT-code/")

# Instantiate the model
autoencoder = Autoencoder(device)

# Create a DataLoader for the tokenized AI-generated data
ai_tokens = torch.cat(tokens)  # Concatenate token tensors
# Split the data into training and validation sets
train_data, val_data = train_test_split(ai_tokens, test_size=0.2)

# Create DataLoaders for both sets
train_loader = DataLoader(TensorDataset(train_data), batch_size=16, shuffle=True)
val_loader = DataLoader(TensorDataset(val_data), batch_size=16, shuffle=False)

# Define optimizer and loss function
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-4, weight_decay=1e-5)  # Reduced learning rate
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
loss_fn = nn.MSELoss()

# Training loop
autoencoder.train()
# Training loop with increased epochs
for epoch in range(50):  # Increased epoch count
    autoencoder.train()
    train_loss = 0.0  # Reset train loss for each epoch
    for batch in train_loader:
        inputs = batch[0].to(device).float()
        optimizer.zero_grad()
        outputs = autoencoder(inputs)
        loss = loss_fn(outputs, inputs)  # Minimize reconstruction error
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Calculate average train loss
    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Average Train Loss: {avg_train_loss}")

    # Adjust learning rate based on the scheduler
    scheduler.step()

    # Evaluate on validation set
    val_loss = 0.0
    autoencoder.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for val_batch in val_loader:
            val_inputs = val_batch[0].to(device).float()
            val_outputs = autoencoder(val_inputs)
            val_loss += loss_fn(val_outputs, val_inputs).item()

    # Calculate average validation loss
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}")

# Save the best model
torch.save(autoencoder.state_dict(), "C:/Users/ABHIMANYU/PycharmProjects/cypherhackathon/Saved_model/otherapproach.pth")
print("Model saved.")


def detect_anomalies(code_sample):
    tokenized = tokenizer.encode_plus(
        code_sample,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )['input_ids'].to(device)

    with torch.no_grad():
        reconstructed = autoencoder(tokenized.float())
        loss = loss_fn(reconstructed, tokenized.float())  # Reconstruction error
        return loss.item()


# Example usage with a human-written code sample
human_code_sample = '''
void do_something(int arr[5][10]) {
    printf("%d\n", arr[0][2]);
}
void func() {
    int *ptr = POINTER_TO_SOMEWHERE;
    do_something(ptr);
}
'''

anomaly_score = detect_anomalies(human_code_sample)
print(f"Anomaly Score: {anomaly_score}")

# You can set a threshold for the anomaly score to classify the code
threshold = 0.05  # Example threshold, adjust based on validation results

if((anomaly_score >100000000) and (anomaly_score< 200000000.0)):
    print("Detected as Human-Written Code")
else:
    print("Detected as AI-Generated Code")

# refactor variable names for clarity

# add detailed comments to complex logic