from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch

# Load the pre-trained CodeBERT model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base")

# Ensure the model is in evaluation mode
model.eval()


# Function to calculate perplexity
def calculate_perplexity(code):
    tokens = tokenizer(code, return_tensors="pt")
    with torch.no_grad():
        output = model(**tokens)
        logits = output.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = tokens.input_ids[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    perplexity = torch.exp(loss.mean())
    return perplexity.item()


'''
public class TwoSum {
    public static int[] twoSum(int[] nums, int target) {
        HashMap<Integer, Integer> numToIndex = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int complement = target - nums[i];
            if (numToIndex.containsKey(complement)) {
                return new int[] { numToIndex.get(complement), i };
            }
            numToIndex.put(nums[i], i);
        }
        return new int[0];
    }
}
'''


# Function to detect if the code is AI-generated or human-generated
def detect_ai_generated_code(code_sample, human_code_sample):
    human_perplexity = calculate_perplexity(human_code_sample)
    ai_perplexity = calculate_perplexity(code_sample)

    print(f"Human Perplexity: {human_perplexity}, AI Code Perplexity: {ai_perplexity}")

    # Define a threshold for perplexity; adjust based on experimentation
    threshold = human_perplexity * 2  # Example threshold (1.5 times the human perplexity)

    if ai_perplexity >threshold:
        return "Human-Generated Code"
    else:
        return "AI-Generated Code"


# Function to read code from a text file
def read_code_from_file(file_path):
    with open(file_path, 'r') as file:
        code = file.read()  # Read the entire file content
    return code