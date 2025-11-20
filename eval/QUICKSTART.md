# AlpacaEval Evaluation Guide

This directory contains scripts to evaluate model outputs using AlpacaEval.

## Prerequisites

1. **Install AlpacaEval**
   ```bash
   pip install alpaca-eval
   ```

2. **Set OpenAI API Key** (required for the evaluator):
   ```bash
   export OPENAI_API_KEY=your_key_here
   ```

3. **Generate model outputs** (if not already done):
```bash
python eval/generate_model_outputs.py --batch_size 128
```
For testing:
```bash
python eval/generate_model_outputs.py --batch_size 128 --max_examples
```

This creates:
- `eval_outputs/base_model_outputs.json`
- `eval_outputs/sft_model_outputs.json`
- `eval_outputs/dpo_model_outputs.json`
- `eval_outputs/adversarial_dpo_model_outputs.json`


## Running Evaluations


Simply run the bash script:

```bash
bash eval/run_alpaca_eval.sh
```

## Comparisons Performed

The script runs 4 comparisons:

1. **base vs sft**: Shows SFT improvement over base model
   - Expected: SFT should have higher win rate (>50%)

2. **sft vs dpo**: Shows DPO improvement over SFT
   - Expected: DPO should have higher win rate (>50%)

3. **dpo vs adversarial_dpo**: Shows adversarial DPO degradation
   - Expected: Adversarial DPO should have lower win rate (<50%)

4. **sft vs adversarial_dpo**: Checks if adversarial DPO is at least better than no DPO
   - Expected: Adversarial DPO should have lower win rate (<50%), but this confirms it's worse than SFT

## Results

Results are saved in `./eval_results/`:

```bash
ls -la eval_results/*/leaderboard.csv
```

Each comparison has its own directory with:
- `leaderboard.csv` - Win rates and statistics
- `annotations.json` - Detailed annotations


## Interpreting Results

- **Win Rate**: Percentage of times the model is preferred over the reference
  - \>50%: Model is better than reference
  - <50%: Model is worse than reference
  - 50%: Models are equivalent

- **Std Error**: Standard error of the win rate estimate

- **Length-controlled Win Rate**: Win rate adjusted for output length bias (if available)
