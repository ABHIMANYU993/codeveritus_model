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
# model.load_state_dict(torch.load('E:/python-projects/llm/Trained_models/codebert_codeparrotjsontrained1.0.pth'))
model.load_state_dict(torch.load('E:/python-projects/llm/Trained_models/codebert_primary.pth',weights_only=True))

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
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

    return probabilities.cpu().numpy()  # Return predictions as numpy array


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

                if sample_list:
                    # Make predictions using the model
                    probabilities = predict_code_samples(model, sample_list)

                    # Print results with percentages for each code sample
                    for idx, (code, prob) in enumerate(zip(sample_list, probabilities)):
                        ai_generated_prob = prob[1] * 100  # Percentage for AI-generated class (label 1)
                        human_generated_prob = prob[0] * 100  # Percentage for Human-written class (label 0)
                        if ai_generated_prob > human_generated_prob:
                            prediction_labels = [f"{ai_generated_prob:.2f}% Of code similar to AI-generated code."]
                        else:
                            prediction_labels = [
                                f"{human_generated_prob:.2f}% Of code similar to Human-generated code."]
                    # prediction_labels = ["AI-generated" if pred == 1 else "Human-generated" for pred in predictions]
                    print(f"Predictions for user {userId}: {prediction_labels}")

                    # Make sure new_data contains both 'processed' and 'processedAt'
                    new_data = {
                        'processed': True,  # Set 'processed' to True after predictions are made
                        'prediction': prediction_labels,  # Store the predictions
                        'processedAt': time.strftime("%Y-%m-%d %H:%M:%S")  # Current timestamp for 'processedAt'
                    }

                    # Update the document in MongoDB
                    result = user_codes_collection.update_one(
                        {'userId': userId},  # Filter to match the document by _id
                        {'$set': new_data}  # Set the 'processed' and 'processedAt' fields
                    )

                    # Check if the update was successful
                    if result.matched_count > 0:
                        print(f"Document updated for userId: {userId}")
                    else:
                        print(f"No document found for userId: {userId}")
        else:
            print("No new submissions found. Waiting...")

    except pymongo.errors.ConnectionFailure as e:
        print("Could not connect to MongoDB:", e)
    except pymongo.errors.OperationFailure as e:
        print(f"Authentication failed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Wait before checking the database again (interval in seconds)
    time.sleep(1)
# update tensorboard logging interval

# comment out experimental code block