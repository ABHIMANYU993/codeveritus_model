# Codeveritus Dataset Preprocessing & Sourcing Notes

## Sourcing Methodology
To compile a robust dataset that represents human developers and multiple generative AI engines, we collected data from two primary channels:
1. **Primary Dataset (`primary_dataset.jsonl`)**: Compiled from open-source GitHub repositories containing files in Python, JavaScript, and Java. These code repositories represent human developers committing production-quality code.
2. **Augmented Dataset (`augmented_dataset.jsonl`)**: Built by prompting various generative LLMs (including ChatGPT, Claude, and Llama) to generate solutions to matching coding prompts. We used diverse prompts to represent various coding styles, commenting styles, and naming conventions.

## Dataset Statistics
Below is the breakdown of code samples in our combined dataset:

| Dataset File | Total Samples | Human Classes | AI Classes | Average Token Count | Language Share (PY/JS/JV) |
|---|---|---|---|---|---|
| `primary_dataset.jsonl` | 12,450 | 12,450 | 0 | 345 | 45% / 35% / 20% |
| `augmented_dataset.jsonl`| 15,200 | 0 | 15,200 | 280 | 40% / 40% / 20% |
| Combined Dataset | 27,650 | 12,450 | 15,200 | 310 | 42.2% / 37.8% / 20% |

## Tokenizer Analysis and Custom Token Checker
We implement custom token checking scripts (`src/token_checking/tokenchecker.py`) to analyze the vocabulary distribution. AI-generated code and human code exhibit distinct differences in token density and keyword usage:
- **AI-Generated Code characteristics**:
  - Heavy use of standard, verbose commenting structures.
  - Repetitive identifier names (e.g. `helper`, `result`, `temp`).
  - High structural uniformity: strict adherence to linting standards.
- **Human-Written Code characteristics**:
  - Sparse, idiomatic comments, often containing colloquial phrases.
  - High diversity in identifier naming, including domain-specific terminology.
  - Inconsistent formatting and stylistic quirks.

The custom token checker scans code files, tokenizes them, and outputs statistics regarding the ratio of comments, average line length, and indentation variance to assist the neural network's feature discovery.

## Detailed Preprocessing Pipeline
The raw text of source code files undergoes several stages of sanitization and processing before being converted to inputs for CodeBERT:

### 1. File Ingestion
The pipeline reads JSONL files line by line, parsing the stringified JSON into python dictionaries containing `code` and `label` (where `0` is Human and `1` is AI).

### 2. Syntax Cleaning
We remove non-ASCII characters and clean up excessive blank lines. This ensures the tokenizer does not waste sequence slots on long blocks of carriage returns.

### 3. Subword Tokenization
We use Byte-Pair Encoding (BPE) pre-trained on code. BPE splits unknown keywords into subwords, preventing out-of-vocabulary errors. For instance, a function like `calculate_fibonacci_sum` might be split into `calculate`, `_fibonacci`, and `_sum`.

### 4. Special Tokens Mapping
Every sequence is prepended with `[CLS]` (classification token) and appended with `[SEP]` (separation token). These tokens notify the CodeBERT transformer of the start and end of the code segment.

### 5. Sequence Masking and Padding
To normalize all inputs to 512 dimensions:
- Sequences longer than 512 tokens are truncated.
- Sequences shorter than 512 are padded using `[PAD]` (index 1).
- An attention mask list of 1s and 0s is created, where 1 indicates an active code token and 0 indicates a padded token.

## Experimental Evaluation on Token Distributions
Our data pipeline performs checks on token distributions to ensure there is no model bias. We visualize the token sequence lengths across classes to verify that the model is not simply classifying code based on sequence length. The average sequence lengths for both classes remain highly comparable, forcing the model to rely on syntax structure and semantics rather than simple document length metrics.
