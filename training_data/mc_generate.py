#!/usr/bin/env python3
"""
Monte Carlo proof completion using vLLM.

Given the output JSONL from inference_steps_vllm.py (which includes cot_steps),
for each theorem and for steps 1..4, build cumulative context and ask the model
to complete the Lean 4 proof. Generate 12 samples per step using vLLM (n=12).

Result: For M theorems, produce M x 4 x 12 completions to be verified later.
"""

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
# Prompt construction
# -----------------------------

def build_problem_prompt(item: Dict[str, Any]) -> str:
    """Construct the base problem prompt from theorem data.
    Prefers informal_prefix if present, always includes formal_statement.
    """
    informal = item.get('informal_prefix', '').strip()
    formal = item.get('formal_statement', '').strip()

    header = "You are an expert in Lean 4 theorem proving."
    parts = [header]
    if informal:
        parts.append("Problem (Natural Language):\n" + informal)
    if formal:
        parts.append("Formal Statement (Lean 4):\n```lean4\n" + formal + "\n```")
    parts.append("Your task: Complete the Lean 4 formal proof for the theorem.")
    return "\n\n".join(parts)


def build_step_context_prompt(base_prompt: str, step_index: int, cot_steps: List[str]) -> str:
    """Build cumulative context for step k (1-based index expected from spec: steps 1..4).
    Input = base problem prompt + steps[0:k] content + instruction to output only Lean code.
    """
    k = step_index
    prev = []
    for i in range(k):
        step_text = (cot_steps[i] or "").strip()
        if step_text:
            prev.append(f"Step {i+1}:\n{step_text}")
    prev_block = "\n\n".join(prev)

    instruction = (
        "Now, using the above reasoning, produce the complete Lean 4 proof code only.\n"
        "Requirements:\n"
        "- Output only Lean 4 code fenced in triple backticks.\n"
        "- Ensure the code compiles in Lean 4.\n"
        "- Do not include any additional commentary.\n"
        "- No placeholders like `sorry`. Provide a full proof.\n"
    )

    final_prompt = base_prompt
    if prev_block:
        final_prompt += "\n\nCumulative Reasoning Steps:\n" + prev_block
    final_prompt += "\n\n" + instruction
    return final_prompt


# -----------------------------
# Core generation
# -----------------------------

def generate_mc_proof_completions(
    model: LLM,
    items: List[Dict[str, Any]],
    samples_per_step: int = 12,
    steps_to_use: int = 4,
    max_tokens: int = 40960,
    temperature: float = 1.0,
    top_p: float = 0.95,
    batch_prompts: int = 8,
) -> List[Dict[str, Any]]:
    """For each item, for steps 1..steps_to_use, generate N completions using cumulative context.
    Returns a list of enhanced items with mc_proof_completions keyed by step.
    """

    def _normalize_cot_steps(item: Dict[str, Any]) -> List[str]:
        """Return exactly 5 CoT steps as a list if present and non-empty, else [].
        Accepts either list-like [step1..step5] or dict-like {step_1:..., ..., step_5:...}.
        A step is considered valid if it is a non-empty string after strip().
        """
        raw = item.get('cot_steps')
        steps: List[str] = []
        if isinstance(raw, list):
            steps = [str(s or '').strip() for s in raw]
        elif isinstance(raw, dict):
            steps = [str(raw.get(f'step_{i}', '') or '').strip() for i in range(1, 6)]
        else:
            steps = []
        # Require exactly 5 non-empty steps
        if len(steps) != 5:
            return []
        if any(not s for s in steps):
            return []
        return steps

    # Prepare all prompts (only for items with all 5 valid steps)
    prompt_records: List[Dict[str, Any]] = []
    valid_item_indices: List[int] = []
    skipped_no_cot: int = 0
    for idx, item in enumerate(items):
        steps5 = _normalize_cot_steps(item)
        if not steps5:
            skipped_no_cot += 1
            continue
        valid_item_indices.append(idx)
        base_prompt = build_problem_prompt(item)
        for s in range(1, steps_to_use + 1):
            prompt = build_step_context_prompt(base_prompt, s, steps5)
            prompt_records.append({
                'item_index': idx,
                'step_index': s,  # 1..steps_to_use
                'prompt': prompt,
            })

    print(
        f"Prepared {len(prompt_records)} prompts for {len(valid_item_indices)} valid items "
        f"(steps={steps_to_use}, skipped_items_without_5_steps={skipped_no_cot})"
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=samples_per_step,
    )

    # Run vLLM in prompt batches, collect n samples each
    outputs_by_item: Dict[int, Dict[int, List[str]]] = {}
    batches = [prompt_records[i:i + batch_prompts] for i in range(0, len(prompt_records), batch_prompts)]

    for b_idx, batch in enumerate(tqdm(batches, desc="Generating MC proofs")):
        prompts = [rec['prompt'] for rec in batch]
        vouts = model.generate(prompts, sampling_params)

        for rec, out in zip(batch, vouts):
            texts = [o.text.strip() for o in out.outputs]
            iidx = rec['item_index']
            sidx = rec['step_index']
            outputs_by_item.setdefault(iidx, {})[sidx] = texts

        if (b_idx + 1) % 10 == 0:
            print(f"Processed {b_idx + 1}/{len(batches)} batches")

    # Attach to original items
    enhanced: List[Dict[str, Any]] = []
    for idx, item in enumerate(items):
        step_to_texts = outputs_by_item.get(idx, {})
        new_item = item.copy()
        new_item['mc_proof_completions'] = {
            f'step_{s}': step_to_texts.get(s, []) for s in range(1, steps_to_use + 1)
        }
        new_item['mc_params'] = {
            'samples_per_step': samples_per_step,
            'steps_to_use': steps_to_use,
            'generation': {
                'max_new_tokens': max_tokens,
                'temperature': temperature,
                'top_p': top_p,
            }
        }
        enhanced.append(new_item)

    return enhanced


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Monte Carlo proof completion with vLLM")
    ap.add_argument('--input_jsonl', type=str, required=True, help='Input JSONL from inference_steps_vllm.py')
    ap.add_argument('--output_jsonl', type=str, required=True, help='Output JSONL with MC completions')
    ap.add_argument('--model_path', type=str, default='Goedel-LM/Goedel-Prover-V2-8B', help='Model id or path')
    ap.add_argument('--samples_per_step', type=int, default=12, help='Num samples per step (n in vLLM)')
    ap.add_argument('--steps_to_use', type=int, default=4, help='Use first K steps (exclude final)')
    ap.add_argument('--max_new_tokens', type=int, default=40960, help='Max tokens to generate')
    ap.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature (match inference.py)')
    ap.add_argument('--top_p', type=float, default=0.95, help='Top-p (match inference.py)')
    ap.add_argument('--batch_prompts', type=int, default=8, help='Num prompts per vLLM call')
    ap.add_argument('--gpu', type=int, default=1, help='Tensor parallel size')
    ap.add_argument('--max_model_len', type=int, default=40960, help='Max model length')
    ap.add_argument('--seed', type=int, default=42, help='Random seed')
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
        steps_to_use=args.steps_to_use,
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
