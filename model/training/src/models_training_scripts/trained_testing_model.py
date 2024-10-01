import torch
from torch import nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import matplotlib.pyplot as plt
from safetensors.torch import load_file


# Step 1: Define the CodeBERT model with Dropout
class CodeBERTClassifier(nn.Module):
    def __init__(self):
        super(CodeBERTClassifier, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)
        self.dropout = nn.Dropout(p=0.25)  # Add a dropout layer

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = self.dropout(outputs.logits)  # Apply dropout
        return logits


# Step 2: Load the trained model
model = CodeBERTClassifier()
model.load_state_dict(load_file('E:/python-projects/models_training/Novathon-JSS/model.safetensors'),strict=False)
# model.load_state_dict(
    # torch.load('E:/python-projects/llm/Trained_models/codebert_codeparrotjsontrained0.9886363636363636.pth'))
# model.load_state_dict(torch.load('E:/python-projects/llm/Trained_models/codebert_primary.pth'))

# model.load_state_dict(load_file('E:/python-projects/models_training/Novathon-JSS/codebert_json10-0.9587301587301588.safetensors'),strict=False)

# model.load_state_dict(torch.load('E:/python-projects/llm/Trained_models/codebert_codeparrotjsontrained1.0.pth'))

model.eval()  # Set the model to evaluation mode

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

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

        # # Find the index of the highest logit
        # max_index = torch.argmax(outputs).item()
        # print("Index of highest logit:", max_index)  # Output: 3
        #
        # # # Find the value of the highest logit
        # # max_logit = outputs[max_index].numpy().item()
        # # print("Highest logit value:", max_logit)  # Output: 3.5
        #
        # # # Optional: Find top-k logits and their indices
        # k = 2
        # top_k_logits, top_k_indices = torch.topk(outputs, k)
        # print("Top-k logits:", top_k_logits.tolist())
        # print("Top-k indices:", top_k_indices.tolist())

    return probabilities.cpu().numpy()
    #     _, preds = torch.m(outputs, dim=1)

    # return preds.cpu().numpy()  # Return predictions as numpy array


# # Load the text file with multiple code snippets
# with open("sample_code.txt", "r") as file:
#     file_content = file.read()

# Split based on delimiter '###'
# code_samples = file_content.split("####")
# code_samples = [code.strip() for code in code_samples if code.strip()]  # Clean up

def format_code_sample(code):
    # Remove leading/trailing whitespaces and split into lines
    lines = code.strip().split("\n")
    # Strip each line and join with a single space
    formatted_code = " ".join(line.strip() for line in lines)
    # Replace multiple spaces with a single space
    formatted_code = " ".join(formatted_code.split())
    return formatted_code


code_samples = [
    """class Solution:
    def findMaximumElegance(self, items: List[List[int]], k: int) -> int:
        items = sorted(items, key=lambda v: -v[0])
        res = cur = 0
        A = []
        seen = set()
        for i, (p, c) in enumerate(items):
            if i < k:
                if c in seen:
                    A.append(p)
                cur += p
            elif c not in seen:
                if not A: break
                cur += p - A.pop()
            seen.add(c)
            res = max(res, cur + len(seen) * len(seen))
        return res
"""
]
# # Make predictions
# predictions = predict_code_samples(model,code_samples)
# print(predictions)

# # Print results
# # for code, prediction in zip(new_code_samples, predictions):
# for idx, (code, prediction) in enumerate(zip(code_samples, predictions)):
#     label = "AI-generated" if prediction == 1 else "Human-generated"
#     print(f"Sample {idx+1}:")
#     print(f"Prediction: {label}\n")
# for_code= [format_code_sample(code_samples)]
print(code_samples)


# def classify_code_lines(model, code_samples):
#     results = []  # Store results for all code samples
#
#     for idx, code_sample in enumerate(code_samples):
#         lines = code_sample.strip().split("\n")  # Split each sample into lines
#         sample_results = []  # Store results for this code sample
#
#         for line in lines:
#             if line.strip():  # Skip empty lines
#                 formatted_line = format_code_sample(line)  # Format the line
#                 probabilities = predict_code_samples(model, [formatted_line])  # Predict single line
#                 ai_generated_prob = probabilities[0][1] * 100  # AI-generated probability
#                 sample_results.append((line, ai_generated_prob))  # Append line with its probability
#
#         results.append(sample_results)  # Append results of this code sample to the main list
#
#     return results


# ai_probabilities_by_sample = classify_code_lines(model, code_samples)

# for line, ai_prob in ai_probabilities_by_sample:
#     print(f"Line: {line}")
#     print(f"AI-generated probability: {ai_prob:.2f}%\n")

# Make predictions
probabilities = predict_code_samples(model, code_samples)

# Print results with percentages for each code sample
for idx, (code, prob) in enumerate(zip(code_samples, probabilities)):
    ai_generated_prob = prob[1] * 100  # Percentage for AI-generated class (label 1)
    human_generated_prob = prob[0] * 100  # Percentage for Human-written class (label 0)
    print(f"Sample {idx + 1}:")
    print(f"Prediction: {ai_generated_prob:.2f}% AI-generated, {human_generated_prob:.2f}% Human-generated\n")

# for sample_idx, sample_results in enumerate(ai_probabilities_by_sample):
#     print(f"Sample {sample_idx + 1}:")
#     for line, ai_prob in sample_results:
#         if ai_prob > 50:  # AI-generated threshold
#             print(f"[AI] {line} ({ai_prob:.2f}%)")
#         else:
#             print(f"[Human] {line} ({ai_prob:.2f}%)")
#     print("-" * 40)
# optimize data loading loop

# update docstrings for better readability