# Multimodal Claim Extraction for Fact-Checking

## Usage

### Running Experiments

```bash
# Baseline experiments
python experiments/baseline_script.py --client openrouter --model google/gemini-2.0-flash-001

# MICE experiments
python experiments/my_approach_script.py --include-vision --client openrouter --model google/gemini-2.0-flash-001

```

### Running Evaluations

```bash
# LLM-based evaluation
python evaluation/llm_evaluation.py --file experiment_results/my_approach_google_gemini-2.0-flash-001.json --eval reference_based --client openrouter --model google/gemini-2.5-flash-lite
```

## Requirements

- Python 3.8+
- Required packages: see `requirements.txt` in parent directory
- API keys for OpenRouter, OpenAI, and Azure Vision services
- Put data in `data/averimatec_with_fulltext/` folder

