import time
import urllib
import pymongo
import torch
from torch import nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from pymongo import MongoClient


# Step 1: Define the CodeBERT model with Dropout
class CodeBERTClassifier(nn.Module):
    def __init__(self):
        super(CodeBERTClassifier, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained(
            "microsoft/codebert-base", num_labels=2
        )
        self.dropout = nn.Dropout(p=0.3)  # Add a dropout layer

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = self.dropout(outputs.logits)  # Apply dropout
        return logits


# Step 2: Load the trained model
model = CodeBERTClassifier()
try:
    model.load_state_dict(
        torch.load('C:/Users/kkvis/OneDrive/Desktop/Trained_models/codebert_model.pth',
                   map_location=torch.device('cpu')))
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: Model file not found. Please check the path.")

model.eval()  # Set the model to evaluation mode

# Use GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

# Step 3: Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")


# # Step 4: Connect to MongoDB and retrieve code samples
# client = MongoClient("mongodb://localhost:27017/")  # Connect to MongoDB
# db = client["qpcode"]  # Access the database
# collection = db["usercodes"]  # Access the collection

# def get_code_samples_from_mongodb():
#     """Fetch all code samples and their IDs from MongoDB."""
#     return [{"_id": doc["_id"], "code": doc["codes"]} for doc in collection.find()]

# Step 5: Preprocess the input code samples
def preprocess_input_code(code_samples):
    tokenized_samples = []
    attention_masks = []

    for code_sample in code_samples:
        tokenized_input = tokenizer(
            code_sample,