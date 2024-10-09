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