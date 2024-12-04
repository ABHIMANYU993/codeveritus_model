import os
import torch
from torch import nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)


# Step 1: Define the CodeBERT model with Dropout
class CodeBERTClassifier(nn.Module):
    def __init__(self):
        super(CodeBERTClassifier, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)
        self.dropout = nn.Dropout(p=0.3)  # Add a dropout layer

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = self.dropout(outputs.logits)  # Apply dropout
        return logits


# Step 2: Download the model from Google Drive if not present
def download_model_from_gdrive(url, destination):
    # Google Drive file ID extraction
    file_id = url.split('/')[-2]
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    response = requests.get(download_url, stream=True)
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded the model file to {destination}")
    else:
        print("Failed to download the model file.")


# Google Drive download link
gdrive_url = "https://drive.google.com/file/d/1FeF4C07z0kJM-9t8ra80WtU37UiiUHCe/view?usp=drive_link"
model_path = "codebert_model.pth"

# Check if model file exists, if not, download it
if not os.path.exists(model_path):
    download_model_from_gdrive(gdrive_url, model_path)

# Step 3: Load the trained model
model = CodeBERTClassifier()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Move model to GPU if available
device = torch.device('cpu')
model.to(device)

# Step 4: Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")


# Step 5: Preprocess the input code samples
def preprocess_input_code(code_samples):
    tokenized_samples = []
    attention_masks = []

    for code_sample in code_samples:
        tokenized_input = tokenizer(
            code_sample,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        tokenized_samples.append(tokenized_input['input_ids'].squeeze(0))
        attention_masks.append(tokenized_input['attention_mask'].squeeze(0))

    tokens = torch.stack(tokenized_samples)
    masks = torch.stack(attention_masks)

    return tokens, masks


# Step 6: Make predictions
def predict_code_samples(model, code_samples):
    tokens, masks = preprocess_input_code(code_samples)

    tokens = tokens.to(device)
    masks = masks.to(device)

    with torch.no_grad():
        outputs = model(tokens, attention_mask=masks)
        _, preds = torch.max(outputs, dim=1)

    return preds.cpu().numpy()


# Step 7: Define the API endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    code_samples = data.get("code_samples", [])

    if not code_samples:
        return jsonify({"error": "No code samples provided"}), 400

    # Make predictions using the model
    predictions = predict_code_samples(model, code_samples)
    prediction_labels = ["AI-generated" if pred == 1 else "Human-generated" for pred in predictions]

    return jsonify({"predictions": prediction_labels})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)