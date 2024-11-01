import torch
from torch import nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import pymongo
from pymongo import MongoClient
import urllib.parse
import time


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


# Step 2: Load the trained model
model = CodeBERTClassifier()
model.load_state_dict(torch.load('E:/python-projects/llm/Trained_models/codebert_model.pth'))
model.eval()  # Set the model to evaluation mode

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Step 3: Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")


# Step 4: Preprocess the input code samples
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

    # Convert to PyTorch tensors
    tokens = torch.stack(tokenized_samples)
    masks = torch.stack(attention_masks)

    return tokens, masks


# Step 5: Make predictions
def predict_code_samples(model, code_samples):
    tokens, masks = preprocess_input_code(code_samples)

    # Move input tensors to the same device as the model
    tokens = tokens.to(device)
    masks = masks.to(device)

    with torch.no_grad():
        outputs = model(tokens, attention_mask=masks)
        _, preds = torch.max(outputs, dim=1)

    return preds.cpu().numpy()  # Return predictions as numpy array


# MongoDB connection
username = urllib.parse.quote_plus("os.getenv('DB_USER', 'user')")
password = urllib.parse.quote_plus("os.getenv('DB_PASS', 'pass')")
connection_string = f"mongodb+srv://{username}:{password}@cluster0.e2ck1.mongodb.net/backend?retryWrites=true&w=majority"
client = MongoClient(connection_string)
db = client['backend']  # Your actual database name
user_codes_collection = db['usercodes']  # Your collection name

# MongoDB query to get unprocessed submissions
while True:
    try:
        # Fetch submissions where 'processed' is False
        user_codes_list = list(user_codes_collection.find({"processed": False}))

        if user_codes_list:
            print("Processing new or updated submissions...")

            for user_code in user_codes_list:
                print(f"Processing userId: {user_code['userId']}")

                userId = user_code['userId']
                codes = user_code.get('codes', [])
                sample_list = [code for code in codes]