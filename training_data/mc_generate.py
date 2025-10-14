#!/usr/bin/env python3
"""
Monte Carlo proof completion using vLLM.

Given the output JSONL from the variable CoT inference script (which includes cot_steps),
for each theorem and for each step, build cumulative context and ask the model
to complete the Lean 4 proof. Generate N samples per step using vLLM.

Key updates:
- Works with variable-length steps (not fixed to 5)
- Enforces Lean4 code block format in output
- Extracts and validates generated code
- Supports multiple samples from original inference
"""

import json
import argparse
import re
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import random

from vllm import LLM, SamplingParams


# -----------------------------
# IO Utilities
# -----------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def save_jsonl(path: str, items: List[Dict[str, Any]]):
    with open(path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


# -----------------------------
# Code extraction
# -----------------------------

def extract_lean4_code(text: str) -> List[str]:
    """Extract all Lean4 code blocks from text.
    
    Looks for both ```lean4 and ```lean markers.
    Returns a list of code strings (without the backticks).
    """
    code_block_regex = re.compile(
        r'```(?:lean4|lean)\s*\n(.*?)```',
        re.DOTALL | re.IGNORECASE
    )
    matches = code_block_regex.findall(text)
    return [match.strip() for match in matches]


# -----------------------------
# Prompt construction
# -----------------------------

def build_problem_prompt(item: Dict[str, Any]) -> str:
    """Construct the base problem prompt from theorem data.
    Includes both informal and formal statements if available.
    """
    informal = item.get('informal_statement', '').strip()
    formal = item.get('formal_statement', '').strip()

    header = "You are an expert in Lean 4 theorem proving."
    parts = [header]
    
    if informal:
        parts.append("**Problem (Natural Language):**\n" + informal)
    
    if formal:
        parts.append("**Formal Statement (Lean 4):**\n```lean4\n" + formal + "\n```")
    
    parts.append("Your task: Complete the Lean 4 formal proof for the theorem.")
    return "\n\n".join(parts)


def build_step_context_prompt(
    base_prompt: str,
    steps: List[Dict[str, str]],
    up_to_step: int
) -> str:
    """Build cumulative context using steps up to (and including) the given step index.
    
    Args:
        base_prompt: Base problem description
        steps: List of step dicts with 'step_num', 'title', 'content', 'lean4_code'
        up_to_step: Include steps up to this step number (inclusive)
    """
    # Filter steps up to the specified step
    relevant_steps = [s for s in steps if s['step_num'] <= up_to_step]
    
    prev = []
    for step in relevant_steps:
        step_num = step['step_num']
        title = step.get('title', '')
        content = step.get('content', '').strip()
        
        if content:
            header = f"### Step {step_num}"
            if title:
                header += f": {title}"
            prev.append(f"{header}\n{content}")
    
    prev_block = "\n\n".join(prev)

    instruction = (
        "\n**Now, using the reasoning above, produce the complete Lean 4 proof.**\n\n"
        "**Requirements:**\n"
        "- Output ONLY Lean 4 code wrapped in triple backticks with the lean4 identifier:\n"
        "  ```lean4\n"
        "  -- Your proof here\n"
        "  ```\n"
        "- Ensure the code compiles in Lean 4\n"
        "- Provide a complete, working proof\n"
        "- NO placeholders like `sorry`\n"
        "- NO additional commentary outside the code block\n"
    )

    final_prompt = base_prompt
    if prev_block:
        final_prompt += "\n\n**Cumulative Reasoning Steps:**\n" + prev_block
    final_prompt += instruction
    return final_prompt


# -----------------------------
# Step normalization
# -----------------------------

def normalize_cot_steps(item: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    """Extract and normalize CoT steps from an item.
    
    Returns a list of step dictionaries if valid, None otherwise.
    Handles both the new format (list of dicts) and legacy format.
    """
    steps = item.get('cot_steps')
    
    if not steps:
        return None
    
    # New format: list of dicts with step_num, title, content, lean4_code
    if isinstance(steps, list) and len(steps) > 0:
        if isinstance(steps[0], dict):
            # Validate that steps have required fields
            valid_steps = []
            for step in steps:
                if 'step_num' in step and 'content' in step:
                    content = step.get('content', '').strip()
                    if content:  # Only include non-empty steps
                        valid_steps.append(step)
            return valid_steps if valid_steps else None
        
        # Legacy format: list of strings
        else:
            valid_steps = []
            for i, content in enumerate(steps, start=1):
                content_str = str(content or '').strip()
                if content_str:
                    valid_steps.append({
                        'step_num': i,
                        'title': '',
                        'content': content_str,
                        'lean4_code': []
                    })
            return valid_steps if valid_steps else None
    
    # Legacy dict format: {step_1: ..., step_2: ..., ...}
    elif isinstance(steps, dict):
        valid_steps = []
        # Try to extract steps in order
        i = 1
        while f'step_{i}' in steps:
            content = steps.get(f'step_{i}', '')
            if isinstance(content, dict):
                # New format nested in dict
                content_str = content.get('content', '').strip()
                if content_str:
                    valid_steps.append(content)
            else:
                # Legacy string format
                content_str = str(content or '').strip()
                if content_str:
                    valid_steps.append({
                        'step_num': i,
                        'title': '',
                        'content': content_str,
                        'lean4_code': []
                    })
            i += 1
        return valid_steps if valid_steps else None
    
    return None


# -----------------------------
# Core generation
# -----------------------------

def generate_mc_proof_completions(
    model: LLM,
    items: List[Dict[str, Any]],
    samples_per_step: int = 12,
    exclude_last_step: bool = True,
    max_tokens: int = 40960,
    temperature: float = 1.0,
    top_p: float = 0.95,
    batch_prompts: int = 8,
) -> List[Dict[str, Any]]:
    """For each item, for each step (optionally excluding last), generate N completions.
    
    Args:
        model: vLLM model instance
        items: List of items with cot_steps
        samples_per_step: Number of completions to generate per step
        exclude_last_step: If True, don't generate for the final step (assumed to already contain proof)
        max_tokens: Max tokens for generation
        temperature: Sampling temperature
        top_p: Top-p sampling
        batch_prompts: Number of prompts to batch in vLLM calls
    """
    # Prepare all prompts
    prompt_records: List[Dict[str, Any]] = []
    valid_item_indices: List[int] = []
    skipped_no_cot: int = 0
    total_prompts_per_item: Dict[int, int] = {}
    
    for idx, item in enumerate(items):
        steps = normalize_cot_steps(item)
        if not steps:
            skipped_no_cot += 1
            continue
        
        valid_item_indices.append(idx)
        base_prompt = build_problem_prompt(item)
        
        # Determine which steps to use for completion
        num_steps = len(steps)
        steps_to_generate = num_steps - 1 if exclude_last_step else num_steps
        
        if steps_to_generate < 1:
            continue
        
        total_prompts_per_item[idx] = steps_to_generate
        
        for step in steps[:steps_to_generate]:
            step_num = step['step_num']
            prompt = build_step_context_prompt(base_prompt, steps, step_num)
            prompt_records.append({
                'item_index': idx,
                'step_num': step_num,
                'prompt': prompt,
                'sample_id': item.get('sample_id', 0),  # Track which sample this is from
            })
    
    print(f"Prepared {len(prompt_records)} prompts for {len(valid_item_indices)} valid items")
    print(f"Skipped {skipped_no_cot} items without valid CoT steps")
    print(f"Generating {samples_per_step} completions per prompt")
    
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=samples_per_step,
    )

    # Run vLLM in prompt batches
    outputs_by_item: Dict[int, Dict[int, List[Dict[str, Any]]]] = {}
    batches = [prompt_records[i:i + batch_prompts] for i in range(0, len(prompt_records), batch_prompts)]
    
    # Track statistics
    total_completions = 0
    completions_with_code = 0

    for b_idx, batch in enumerate(tqdm(batches, desc="Generating MC proofs")):
        prompts = [rec['prompt'] for rec in batch]
        vouts = model.generate(prompts, sampling_params)

        for rec, out in zip(batch, vouts):
            iidx = rec['item_index']
            step_num = rec['step_num']
            
            completions = []
            for sample_idx, vout in enumerate(out.outputs):
                text = vout.text.strip()
                lean4_blocks = extract_lean4_code(text)
                
                total_completions += 1
                if lean4_blocks:
                    completions_with_code += 1
                
                completions.append({
                    'completion_id': sample_idx,
                    'raw_output': text,
                    'lean4_code': lean4_blocks,
                    'has_code': len(lean4_blocks) > 0,
                })
            
            outputs_by_item.setdefault(iidx, {})[step_num] = completions

        if (b_idx + 1) % 10 == 0:
            coverage = completions_with_code / total_completions if total_completions > 0 else 0
            print(f"Processed {b_idx + 1}/{len(batches)} batches | "
                  f"Code coverage: {completions_with_code}/{total_completions} ({coverage:.1%})")

    # Attach to original items
    enhanced: List[Dict[str, Any]] = []
    for idx, item in enumerate(items):
        step_to_completions = outputs_by_item.get(idx, {})
        new_item = item.copy()
        
        # Organize completions by step
        mc_completions = {}
        for step_num, completions in step_to_completions.items():
            mc_completions[f'step_{step_num}'] = completions
        
        new_item['mc_proof_completions'] = mc_completions
        new_item['mc_params'] = {
            'samples_per_step': samples_per_step,
            'num_steps_generated': len(step_to_completions),
            'exclude_last_step': exclude_last_step,
            'generation': {
                'max_new_tokens': max_tokens,
                'temperature': temperature,
                'top_p': top_p,
            }
        }
        enhanced.append(new_item)
    
    # Final statistics
    coverage = completions_with_code / total_completions if total_completions > 0 else 0
    print(f"\n=== Monte Carlo Generation Summary ===")
    print(f"Total completions: {total_completions}")
    print(f"Completions with Lean4 code: {completions_with_code} ({coverage:.1%})")
    print(f"Items processed: {len(valid_item_indices)}")
    print(f"Samples per step: {samples_per_step}")

    return enhanced


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Monte Carlo proof completion with vLLM")
    ap.add_argument('--input_jsonl', type=str, required=True, 
                    help='Input JSONL from variable CoT inference')
    ap.add_argument('--output_jsonl', type=str, required=True, 
                    help='Output JSONL with MC completions')
    ap.add_argument('--model_path', type=str, default='Goedel-LM/Goedel-Prover-V2-8B', 
                    help='Model id or path')
    ap.add_argument('--samples_per_step', type=int, default=12, 
                    help='Number of completions to generate per step')
    ap.add_argument('--exclude_last_step', action='store_true', default=True,
                    help='Exclude the last step from completion generation (assumed to contain final proof)')
    ap.add_argument('--include_last_step', dest='exclude_last_step', action='store_false',
                    help='Include the last step in completion generation')
    ap.add_argument('--max_new_tokens', type=int, default=40960, 
                    help='Max tokens to generate')
    ap.add_argument('--temperature', type=float, default=1.0, 
                    help='Sampling temperature')
    ap.add_argument('--top_p', type=float, default=0.95, 
                    help='Top-p nucleus sampling')
    ap.add_argument('--batch_prompts', type=int, default=8, 
                    help='Number of prompts per vLLM call')
    ap.add_argument('--gpu', type=int, default=1, 
                    help='Tensor parallel size')
    ap.add_argument('--max_model_len', type=int, default=40960, 
                    help='Max model context length')
    ap.add_argument('--seed', type=int, default=42, 
                    help='Random seed')
    args = ap.parse_args()

    random.seed(args.seed)

    print(f"Loading input: {args.input_jsonl}")
    items = load_jsonl(args.input_jsonl)
    print(f"Loaded {len(items)} items")

    print("Loading vLLM model...")
    model = LLM(
        model=args.model_path,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.gpu,
        seed=args.seed,
    )

    enhanced = generate_mc_proof_completions(
        model=model,
        items=items,
        samples_per_step=args.samples_per_step,
        exclude_last_step=args.exclude_last_step,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_prompts=args.batch_prompts,
    )

    print(f"Saving output: {args.output_jsonl}")
    save_jsonl(args.output_jsonl, enhanced)
    print("Done.")


if __name__ == '__main__':
    main()