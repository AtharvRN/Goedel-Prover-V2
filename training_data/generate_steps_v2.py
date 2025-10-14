#!/usr/bin/env python3
"""
Inference script to generate variable-length Chain-of-Thought (CoT) for mathematical theorems using vLLM.

Key features:
- Variable number of CoT steps (not fixed to 5)
- Multiple samples per question
- Clean structure with vLLM for efficient batch generation
- Robust parsing and flexible prompt format
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

def create_cot_prompt(formal_statement: str, informal_statement: str = None) -> str:
    """Create a flexible CoT prompt that allows variable number of steps.
    
    The model should generate a detailed chain of thought with natural step progression.
    Enforces Lean4 code to be in proper code blocks for easy extraction.
    """
    theorem_text = f"**Formal Statement:**\n```lean4\n{formal_statement}\n```"
    if informal_statement:
        theorem_text = f"**Informal Statement:**\n{informal_statement}\n\n" + theorem_text
    
    prompt = f"""You are an expert mathematician working on automated theorem proving. Your task is to solve the following theorem with a detailed chain-of-thought approach.

{theorem_text}

Provide a comprehensive solution with clear reasoning steps. Use markdown headers to organize your solution:
- Use `### Step X: <descriptive title>` format for each step (where X is the step number)
- Include as many steps as needed to thoroughly solve the problem
- Each step should contain detailed reasoning and explanations
- Common steps might include:
  - Problem analysis and restatement
  - Informal mathematical reasoning
  - Strategy formulation
  - Detailed proof construction
  - Solution verification
  - Formal proof in Lean 4

IMPORTANT: When providing Lean 4 code (tactics, proofs, or formal statements), ALWAYS wrap it in triple backticks with the lean4 language identifier:
```lean4
-- Your Lean 4 code here
```

Begin your solution:
"""
    return prompt


def parse_cot_steps(response_text: str) -> List[Dict[str, str]]:
    """Parse variable-length steps using markdown headers '### Step X:' as delimiters.
    
    Returns a list of dictionaries, each containing:
    - 'step_num': the step number
    - 'title': the step title
    - 'content': the step content
    - 'lean4_code': extracted Lean4 code blocks (if any)
    """
    text = (response_text or '').strip()
    if not text:
        return []

    # Match headers like: '### Step 1: ...' at start of line
    header_regex = re.compile(r'(^|\n)###\s*Step\s+(\d+)\s*:\s*([^\n]*)\n', re.IGNORECASE)
    matches = list(header_regex.finditer(text))

    if not matches:
        # If no structured steps found, return the entire text as one step
        lean4_blocks = extract_lean4_code(text)
        return [{
            'step_num': 1,
            'title': 'Solution',
            'content': text,
            'lean4_code': lean4_blocks
        }]

    steps = []
    for i, m in enumerate(matches):
        step_num = int(m.group(2))
        title = m.group(3).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        
        # Extract Lean4 code blocks from this step's content
        lean4_blocks = extract_lean4_code(content)
        
        steps.append({
            'step_num': step_num,
            'title': title,
            'content': content,
            'lean4_code': lean4_blocks
        })

    return steps


def extract_lean4_code(text: str) -> List[str]:
    """Extract all Lean4 code blocks from text.
    
    Looks for both ```lean4 and ```lean markers.
    Returns a list of code strings (without the backticks).
    """
    # Match code blocks with lean4 or lean language identifier
    code_block_regex = re.compile(
        r'```(?:lean4|lean)\s*\n(.*?)```',
        re.DOTALL | re.IGNORECASE
    )
    matches = code_block_regex.findall(text)
    return [match.strip() for match in matches]


# -----------------------------
# Batch Preparation
# -----------------------------

def prepare_records(
    theorems: List[Dict[str, Any]], 
    num_samples: int = 1
) -> List[Dict[str, Any]]:
    """Prepare records with multiple samples per theorem."""
    records = []
    for idx, item in enumerate(theorems):
        formal_statement = item.get('formal_statement', '')
        if not formal_statement:
            continue
        
        informal_statement = item.get('informal_statement', '')
        problem_id = item.get('problem_id', item.get('name', f'item_{idx}'))
        
        # Create multiple samples for the same theorem
        for sample_idx in range(num_samples):
            prompt = create_cot_prompt(formal_statement, informal_statement)
            records.append({
                'idx': idx,
                'sample_idx': sample_idx,
                'problem_id': problem_id,
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
    num_samples: int = 1,
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
    records = prepare_records(raw_items, num_samples=num_samples)
    print(f"Prepared {len(records)} records ({num_samples} samples per item)")
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

    # Aggregate statistics
    total_steps = 0
    total_items = 0
    step_counts = []

    for batch_idx, chunk in enumerate(tqdm(chunks, desc="Generating")):
        prompts = [rec['prompt'] for rec in chunk]
        vllm_out = llm.generate(prompts, sampling_params)
        
        for rec, out in zip(chunk, vllm_out):
            text = out.outputs[0].text.strip()
            steps = parse_cot_steps(text)
            
            num_steps = len(steps)
            step_counts.append(num_steps)
            total_steps += num_steps
            total_items += 1
            
            # Convert steps to both list and dict formats for flexibility
            steps_dict = {f'step_{s["step_num"]}': s for s in steps}
            
            print(f"[{rec['problem_id']}][Sample {rec['sample_idx']}] Generated {num_steps} steps")

            enhanced = rec['source'].copy()
            enhanced.update({
                'sample_id': rec['sample_idx'],
                'cot_prompt': rec['prompt'],
                'cot_response': text,
                'cot_steps': steps,  # List of dicts with step_num, title, content
                'cot_steps_dict': steps_dict,  # Dict keyed by step_num
                'num_steps': num_steps,
                'generation_params': {
                    'max_new_tokens': max_tokens,
                    'temperature': temperature,
                    'top_p': top_p,
                    'num_samples': num_samples,
                },
            })
            outputs.append(enhanced)

        # Save intermediate results every few batches
        if (batch_idx + 1) % 5 == 0:
            inter_path = output_path.replace('.jsonl', f'.part{batch_idx + 1}.jsonl')
            save_jsonl(inter_path, outputs)
            print(f"Saved intermediate results to {inter_path}")

    # Save final
    save_jsonl(output_path, outputs)
    print(f"Saved {len(outputs)} items to {output_path}")

    # Statistics summary
    if total_items > 0:
        avg_steps = total_steps / total_items
        min_steps = min(step_counts) if step_counts else 0
        max_steps = max(step_counts) if step_counts else 0
        print("\n=== CoT Generation Summary ===")
        print(f"Items processed: {total_items}")
        print(f"Unique problems: {len(raw_items)}")
        print(f"Samples per problem: {num_samples}")
        print(f"Average steps per solution: {avg_steps:.2f}")
        print(f"Step count range: {min_steps} - {max_steps}")


# -----------------------------
# CLI
# -----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="vLLM variable-length CoT inference with multiple samples")
    p.add_argument('--input_jsonl', type=str, required=True, help='Path to input JSONL (minif2f-like)')
    p.add_argument('--output_jsonl', type=str, required=True, help='Path to save output JSONL with CoT')
    p.add_argument('--model_path', type=str, default='Goedel-LM/Goedel-Prover-V2-8B', help='HF model id or local path')
    p.add_argument('--num_samples', type=int, default=1, help='Number of CoT samples to generate per problem')
    p.add_argument('--max_new_tokens', type=int, default=40960, help='Max tokens to generate per item')
    p.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    p.add_argument('--top_p', type=float, default=0.95, help='Top-p nucleus sampling')
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
        num_samples=args.num_samples,
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