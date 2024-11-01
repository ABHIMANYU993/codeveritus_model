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
        # Load the pretrained CodeBERT model
        self.model = RobertaForSequenceClassification.from_pretrained(
            "microsoft/codebert-base", num_labels=2
        )
        self.dropout = nn.Dropout(p=0.3)  # Add a dropout layer

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = self.dropout(outputs.logits)  # Apply dropout to the logits
        return logits


# Step 2: Load the trained model
model = CodeBERTClassifier()
try:
    model.load_state_dict(torch.load('E:/python-projects/llm/Trained_models/codebert_model.pth',map_location=torch.device('cpu')))
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: Model file not found. Please check the path.")

model.eval()  # Set the model to evaluation mode

# Use GPU if available, otherwise use CPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
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


# Step 6: Make predictions
def predict_code_samples(model, code_samples):
    tokens, masks = preprocess_input_code([sample["code"] for sample in code_samples])

    # Move input tensors to the same device as the model
    tokens = tokens.to(device)
    masks = masks.to(device)

    with torch.no_grad():
        outputs = model(tokens, attention_mask=masks)
        _, preds = torch.max(outputs, dim=1)

    return preds.cpu().numpy()  # Return predictions as numpy array


# # Step 7: Store predictions back in MongoDB
# def save_predictions_to_mongodb(predictions, code_samples):
#     """Update MongoDB documents with the prediction results."""
#     for prediction, sample in zip(predictions, code_samples):
#         label = "AI-generated" if prediction == 1 else "Human-generated"

#         # Update the MongoDB document with the prediction
#         collection.update_one(
#             {"_id": sample["_id"]},  # Match by document ID
#             {"$set": {"prediction": label}}  # Add/Update 'prediction' field
#         )
#     print("Predictions saved to MongoDB successfully.")

# # Step 8: Get code samples and make predictions
# code_samples = get_code_samples_from_mongodb()

# if code_samples:
#     predictions = predict_code_samples(model, code_samples)

#     # Step 9: Save predictions to MongoDB
#     save_predictions_to_mongodb(predictions, code_samples)

#     # Print predictions for reference
#     for code, prediction in zip(code_samples, predictions):
#         label = "AI-generated" if prediction == 1 else "Human-generated"
#         print(f"Code:\n{code['code']}\nPrediction: {label}\n")
# else:
#     print("No code samples found in MongoDB.")


# MongoDB connection
username = urllib.parse.quote_plus("os.getenv('DB_USER', 'user')")
password = urllib.parse.quote_plus("os.getenv('DB_PASS', 'pass')")
connection_string = f"mongodb+srv://{username}:{password}@cluster0.e2ck1.mongodb.net/backend?retryWrites=true&w=majority"
client = MongoClient(connection_string)
db = client['backend']  # Your actual database name
user_codes_collection = db['usercodes']  # Your collection name

# MongoDB query to get unprocessed submissions
while True: