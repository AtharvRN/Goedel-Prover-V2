#!/usr/bin/env python3
"""
Inference script to generate 5-step Chain-of-Thought (CoT) for minif2f theorems using vLLM.

Design goals:
- Clean structure modeled after Goedel-Prover-V2/src/inference.py
- Uses vLLM for efficient batch generation
- Produces the same output semantics as generate_steps.py (cot_response, cot_steps)
- Robust parsing and clear prompt format
"""

import os
import re
import json
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
import random

from vllm import LLM, SamplingParams


# -----------------------------
# IO Utilities
# -----------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def save_jsonl(path: str, items: List[Dict[str, Any]]):
    with open(path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


# -----------------------------
# Prompt + Parsing
# -----------------------------

def create_cot_prompt(formal_statement: str) -> str:
    """Create a 5-step CoT prompt using markdown headers as the ONLY step delimiters.

    The model must output exactly five sections, each starting with a header:
    '### Step i: <title>' on its own line, for i = 1..5.
    """
    prompt = f"""You are an expert mathematician working on automated theorem proving. Your task is to solve the following Lean 4 theorem step by step.

**Theorem to Prove:**
```lean4
{formal_statement}
```

Output EXACTLY five sections, each clearly demarcated using this markdown header format:
- A header line starting with: `### Step i: <title>` (three hash marks), for i = 1..5
- The content for that step on the following lines

Do NOT use any other block markers or separators. Do NOT produce extra steps.
Use these exact headers (copy verbatim), and place the corresponding content below each:

### Step 1: Problem Analysis and Restatement
[Your content for Step 1]

### Step 2: Informal Mathematical Reasoning
[Your content for Step 2]

### Step 3: Solution Verification
[Your content for Step 3]

### Step 4: Lean 4 Tactics Planning
[Your content for Step 4]

### Step 5: Complete Lean 4 Formal Proof
[Your content for Step 5]
"""
    return prompt


def parse_cot_steps(response_text: str) -> List[str]:
    """Parse 5 steps using ONLY markdown headers '### Step X:' as delimiters.

    We search for lines matching exactly the header pattern and extract the
    content until the next header or end of text. If fewer than 5 are found,
    pad with empty strings. If more are found, keep the first five by index.
    """
    import re

    text = (response_text or '').strip()
    if not text:
        return ["", "", "", "", ""]

    # Match headers like: '### Step 1: ...' at start of line
    header_regex = re.compile(r'(^|\n)###\s*Step\s+(\d+)\s*:[^\n]*\n', re.IGNORECASE)
    matches = list(header_regex.finditer(text))

    steps_map: Dict[int, str] = {}
    for i, m in enumerate(matches):
        step_num = int(m.group(2))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        steps_map[step_num] = content

    # Build ordered list for steps 1..5, padding if missing
    steps: List[str] = [steps_map.get(i, "").strip() for i in range(1, 6)]
    return steps


# -----------------------------
# Batch Preparation
# -----------------------------

def prepare_records(theorems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records = []
    for idx, item in enumerate(theorems):
        # print(item)
        formal_statement = item.get('formal_statement', '')
        if not formal_statement:
            continue
        prompt = create_cot_prompt(formal_statement)
        # print(prompt)
        records.append({
            'idx': idx,
            'problem_id': item.get('problem_id', item.get('name', f'item_{idx}')),
            'prompt': prompt,
            'source': item,
        })
    return records


# -----------------------------
# Inference
# -----------------------------

def run_inference(
    model_path: str,
    inputs_path: str,
    output_path: str,
    max_tokens: int = 40960,
    temperature: float = 1.0,
    top_p: float = 0.95,
    tensor_parallel_size: int = 1,
    max_model_len: int = 8192,
    batch_size: int = 8,
    seed: int = 42,
):
    random.seed(seed)

    # Load data
    raw_items = load_jsonl(inputs_path)
    print(f"Loaded {len(raw_items)} input items")
    records = prepare_records(raw_items)
    print(f"Prepared {len(records)} records with formal statements")
    if not records:
        print("No valid input records; exiting.")
        return

    # Initialize model
    print("Loading vLLM model...")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=1,
    )

    # Batched generation
    outputs: List[Dict[str, Any]] = []
    chunks = [records[i:i + batch_size] for i in range(0, len(records), batch_size)]
    print(f"Processing {len(records)} records in {len(chunks)} batches (batch_size={batch_size})")

    # Aggregate validity stats
    total_non_empty = 0
    total_empty = 0
    total_non_empty_first4 = 0
    total_items = 0

    for batch_idx, chunk in enumerate(tqdm(chunks, desc="Generating")):
        prompts = [rec['prompt'] for rec in chunk]
        # print(prompts)
        vllm_out = llm.generate(prompts, sampling_params)
        # print(vllm_out)
        for rec, out in zip(chunk, vllm_out):

            # print("out", out)
            text = out.outputs[0].text.strip()
            # print("text", text)
            steps = parse_cot_steps(text)
            # print("steps", steps)
            step_dict = {f'step_%d' % (i+1): steps[i] if i < len(steps) else "" for i in range(5)}

            # Per-item validity stats (print only)
            non_empty = sum(1 for s in step_dict.values() if s and s.strip())
            empty = 5 - non_empty
            first4 = [step_dict['step_1'], step_dict['step_2'], step_dict['step_3'], step_dict['step_4']]
            non_empty_first4 = sum(1 for s in first4 if s and s.strip())
            empty_first4 = 4 - non_empty_first4
            print(f"[{rec['problem_id']}] non-empty steps: {non_empty}/5 (first4: {non_empty_first4}/4)")

            enhanced = rec['source'].copy()
            enhanced.update({
                'cot_prompt': rec['prompt'],
                'cot_response': text,
                'cot_steps': step_dict,
                'generation_params': {
                    'max_new_tokens': max_tokens,
                    'temperature': temperature,
                    'top_p': top_p,
                },
            })
            outputs.append(enhanced)

            # Update aggregate stats
            total_items += 1
            total_non_empty += non_empty
            total_empty += empty
            total_non_empty_first4 += non_empty_first4

        # Optional: Save intermittently every few batches
        if (batch_idx + 1) % 5 == 0:
            inter_path = output_path.replace('.jsonl', f'.part{batch_idx + 1}.jsonl')
            save_jsonl(inter_path, outputs)
            print(f"Saved intermediate results to {inter_path}")

    # Save final
    save_jsonl(output_path, outputs)
    print(f"Saved {len(outputs)} items to {output_path}")

    # Aggregate summary
    if total_items > 0:
        avg_non_empty = total_non_empty / (total_items * 5)
        avg_non_empty_first4 = total_non_empty_first4 / (total_items * 4)
        print("\n=== CoT Step Validity Summary ===")
        print(f"Items processed: {total_items}")
        print(f"Avg non-empty ratio (all 5 steps): {avg_non_empty:.2%}")
        print(f"Avg non-empty ratio (first 4 steps): {avg_non_empty_first4:.2%}")


# -----------------------------
# CLI
# -----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="vLLM CoT inference for minif2f")
    p.add_argument('--input_jsonl', type=str, required=True, help='Path to input JSONL (minif2f-like)')
    p.add_argument('--output_jsonl', type=str, required=True, help='Path to save output JSONL with CoT')
    p.add_argument('--model_path', type=str, default='Goedel-LM/Goedel-Prover-V2-8B', help='HF model id or local path')
    p.add_argument('--max_new_tokens', type=int, default=40960, help='Max tokens to generate per item')
    p.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature (match src/inference.py)')
    p.add_argument('--top_p', type=float, default=0.95, help='Top-p nucleus sampling (match src/inference.py)')
    p.add_argument('--gpu', type=int, default=1, help='Tensor parallel size (number of GPUs)')
    p.add_argument('--max_model_len', type=int, default=8192, help='Max model context length')
    p.add_argument('--batch_size', type=int, default=8, help='Batch size of prompts per vLLM call')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    run_inference(
        model_path=args.model_path,
        inputs_path=args.input_jsonl,
        output_path=args.output_jsonl,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        tensor_parallel_size=args.gpu,
        max_model_len=args.max_model_len,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
