#!/bin/bash
set -e

# Configuration
EVAL_OUTPUTS_DIR="./eval_outputs"
RESULTS_DIR="./eval_results"
ANNOTATOR_CONFIG="weighted_alpaca_eval_gpt4_turbo"
OPENAI_MAX_CONCURRENCY="1"  # Set to 1 to minimize rate limits

# Check OPENAI_API_KEY
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY not set!"
    echo "Set it with: export OPENAI_API_KEY=your_key_here"
    exit 1
fi

export OPENAI_API_KEY
export OPENAI_MAX_CONCURRENCY
export PYTHONWARNINGS="ignore"  # Suppress alpaca_eval config warnings
echo "Starting AlpacaEval with concurrency=${OPENAI_MAX_CONCURRENCY}"

# Install if needed
python -c "import alpaca_eval" 2>/dev/null || pip install -q alpaca-eval

mkdir -p "$RESULTS_DIR"

# Function to run evaluation
run_eval() {
    local model=$1 ref=$2 name=$3
    echo -e "\n=== Running: $name (${model} vs ${ref}) ==="

    alpaca_eval evaluate \
        --model_outputs "${EVAL_OUTPUTS_DIR}/${model}_model_outputs.json" \
        --reference_outputs "${EVAL_OUTPUTS_DIR}/${ref}_model_outputs.json" \
        --annotators_config "$ANNOTATOR_CONFIG" \
        --name "$model" \
        --output_path "${RESULTS_DIR}/${name}"

    echo "✓ Saved to ${RESULTS_DIR}/${name}"
}

# Run comparisons
run_eval "sft" "base" "base_vs_sft"
# run_eval "dpo" "sft" "sft_vs_dpo"
# run_eval "adversarial_dpo" "dpo" "dpo_vs_adversarial_dpo"
# run_eval "adversarial_dpo" "sft" "sft_vs_adversarial_dpo"

echo -e "\n✓ All evaluations completed! Results in: ${RESULTS_DIR}"
