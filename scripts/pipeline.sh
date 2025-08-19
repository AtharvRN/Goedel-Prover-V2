#!/bin/bash
# -----------------------------------------------------------------------------
# This is a Bash script to run the Lean proof generation, compilation, and
# summarization pipeline.
#
# Workflow:
#   1. Inference (inference.py): Uses a Large Language Model to generate Lean
#      code proofs from input problems.
#   2. Compilation (compile.py): Compiles the generated Lean code to check
#      for correctness.
#   3. Summarization (summarize.py): Analyzes the compilation results and
#      generates a summary report.
#
# The script supports multiple correction rounds. If a proof generated in one
# round fails, the script automatically proceeds to the next round, feeding the
# error information back to the model for another attempt.
#
# Usage:
#   1. Set all paths and parameters in the "CONFIGURATION" section.
#   2. Run from the terminal: bash run_pipeline.sh
# -----------------------------------------------------------------------------

# Exit immediately if a command fails
set -e

# --- CONFIGURATION ---
# =============================================================================
# *** MODIFY YOUR SETTINGS HERE ***

# --- Model and Data Paths ---
# MODEL_PATH="/path/to/your/llm/model"  # Path to your Large Language Model
# DATA_PATH="path/to/your/input_problems.jsonl" # Path to your input problems file (e.g., minif2f.jsonl)
MODEL_PATH="Goedel-LM/Goedel-Prover-V2-8B"
DATA_PATH="dataset/test.jsonl" # Example path

# --- Output Directory ---
# All generated files (inference results, compilation logs, reports) will be saved here.
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT_DIR="results/run_${TIMESTAMP}"

# --- Inference Settings ---
INFERENCE_HANDLER="dpskcot" # Inference handler, options: "dpskcot", "dpsknoncot", "kiminacot"
GPUS=2                    # Number of GPUs to use for vLLM inference
NUM_SAMPLES_INITIAL=32     # Number of proof samples to generate per problem in the initial round (Round 0)
NUM_SAMPLES_CORRECTION=2  # Number of correction samples to generate per failed attempt in correction rounds (Round > 0)
TEMPERATURE=1.0           # Inference temperature
MAX_MODEL_LEN=40960       # Maximum model sequence length

# --- Compilation Settings ---
CPUS=32                   # Number of CPU cores to use for parallel compilation

# --- Pipeline Control ---
# Maximum number of correction rounds (0 for initial inference only, 1 for initial + one correction round, etc.)
MAX_CORRECTION_ROUNDS=2

# =============================================================================

# Create the output directory
mkdir -p "$BASE_OUTPUT_DIR"
echo "All outputs will be saved to: ${BASE_OUTPUT_DIR}"

# --- Main Loop ---
for round in $(seq 0 $MAX_CORRECTION_ROUNDS); do
    echo
    echo "===================================================="
    echo "===============   Starting Round ${round}   ==============="
    echo "===================================================="

    # --- Step 1: Inference ---
    echo
    echo "--- [Step 1/3] Running Inference (Round ${round}) ---"

    # Set round-specific parameters
    if [ "$round" -eq 0 ]; then
        # Round 0: Create proofs from the initial dataset
        INPUT_ARG="--input_path ${DATA_PATH}"
        PREV_RUN_ARG="" # No previous run output is needed
        NUM_SAMPLES=$NUM_SAMPLES_INITIAL
    else
        # Correction Round (> 0): Correct proofs based on failures from the previous round
        INPUT_ARG="" # Initial dataset is not needed
        # --previous_run_output_dir points to the directory with the previous round's results
        PREV_RUN_ARG="--previous_run_output_dir ${BASE_OUTPUT_DIR}"
        NUM_SAMPLES=$NUM_SAMPLES_CORRECTION
    fi

    # Build and run the inference command
    INFERENCE_CMD="python src/inference.py \
        --model_path ${MODEL_PATH} \
        --output_dir ${BASE_OUTPUT_DIR} \
        --n ${NUM_SAMPLES} \
        --gpu ${GPUS} \
        --inference_handler ${INFERENCE_HANDLER} \
        --correction_round ${round} \
        --max_model_len ${MAX_MODEL_LEN} \
        --trunck 3 \
        --temp ${TEMPERATURE} \
        ${INPUT_ARG} \
        ${PREV_RUN_ARG}"

    echo "Executing command:"
    echo "${INFERENCE_CMD}"
    ${INFERENCE_CMD}

    # Check if the inference output file exists
    SUFFIX=""
    if [ "$round" -gt 0 ]; then
        SUFFIX="_corr${round}"
    fi
    INFERENCE_OUTPUT_FILE="${BASE_OUTPUT_DIR}/to_inference_codes${SUFFIX}.json"
    if [ ! -f "$INFERENCE_OUTPUT_FILE" ]; then
        echo "Error: Inference output file ${INFERENCE_OUTPUT_FILE} not found! Terminating."
        exit 1
    fi
    echo "Inference complete. Output file: ${INFERENCE_OUTPUT_FILE}"

    # --- Step 2: Compilation ---
    echo
    echo "--- [Step 2/3] Running Compilation (Round ${round}) ---"
    
    COMPILE_OUTPUT_FILE="${BASE_OUTPUT_DIR}/code_compilation_repl${SUFFIX}.json"

    # Build and run the compilation command
    COMPILE_CMD="python src/compile.py \
        --input_path ${INFERENCE_OUTPUT_FILE} \
        --output_path ${COMPILE_OUTPUT_FILE} \
        --cpu ${CPUS}"

    echo "Executing command:"
    echo "${COMPILE_CMD}"
    ${COMPILE_CMD}

    # Check if the compilation output file exists
    if [ ! -f "$COMPILE_OUTPUT_FILE" ]; then
        echo "Error: Compilation output file ${COMPILE_OUTPUT_FILE} not found! Terminating."
        exit 1
    fi
    echo "Compilation complete. Output file: ${COMPILE_OUTPUT_FILE}"

    # --- Step 3: Summarization ---
    echo
    echo "--- [Step 3/3] Generating Summary (Round ${round}) ---"

    FULL_RECORDS_FILE="${BASE_OUTPUT_DIR}/full_records${SUFFIX}.json"
    SUMMARY_OUTPUT_DIR="${BASE_OUTPUT_DIR}/summary_round_${round}"
    mkdir -p "$SUMMARY_OUTPUT_DIR"

    # Build and run the summarization command
    SUMMARY_CMD="python src/summarize.py \
        --input_path ${COMPILE_OUTPUT_FILE} \
        --full_record_path ${FULL_RECORDS_FILE} \
        --output_dir ${SUMMARY_OUTPUT_DIR}"

    echo "Executing command:"
    echo "${SUMMARY_CMD}"
    ${SUMMARY_CMD}
    echo "Summary reports generated in: ${SUMMARY_OUTPUT_DIR}"

done

echo
echo "===================================================="
echo "All rounds completed successfully!"
echo "Final results are saved in the directory: ${BASE_OUTPUT_DIR}"
echo "===================================================="