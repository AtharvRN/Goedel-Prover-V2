#!/usr/bin/env python3
"""
DreamPRM-guided inference pipeline for Goedel-Prover.

For each minif2f problem, this script:
1. Generates multiple CoT candidates with the Goedel prover (via vLLM).
2. Scores every candidate using a Process Reward Model (DreamPRM).
   - Only evaluates intermediate reasoning steps (final Lean step excluded).
   - Uses the same conversational template as PRM training.
3. Selects the candidate with the highest mean step score.
4. Compiles only the selected proof with the Lean REPL scheduler.

Outputs a JSON report containing all candidates, scores, and compilation results.
"""

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

# Ensure we can import helper utilities from the repository
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

# Import prompt helpers from training_data/generate_steps.py
from training_data.generate_steps import create_cot_prompt, parse_cot_steps  # type: ignore

# Lean compilation scheduler
from lean_compiler.repl_scheduler import scheduler  # type: ignore


# ---------------------------------------------------------------------------
# Simple prompt and parsing utilities
# ---------------------------------------------------------------------------

def create_simple_prompt(formal_statement: str) -> str:
    return f"""
Complete the following Lean 4 code:

```lean4
{formal_statement}
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
""".strip()


def parse_simple_steps(response: str) -> List[str]:
    """
    Parse the response into reasoning steps based on the simple prompt format.
    Extracts sections starting with '###' as steps.
    """
    lines = response.split('\n')
    steps = []
    current_step = []
    for line in lines:
        line = line.strip()
        if line.startswith("###"):
            if current_step:
                steps.append(' '.join(current_step).strip())
            current_step = [line]
        elif current_step:
            current_step.append(line)
    if current_step:
        steps.append(' '.join(current_step).strip())
    return steps


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# PRM conversation utilities
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an advanced AI assistant, designed to serve as a process supervision model. "
    "In this task, I will provide a problem statement followed by the first step of the solution process. "
    "For each subsequent turn, I will give you a new step in the solution. Your role is to assess "
    "whether the solution process is correct up to the current step.\n\n"
    "- In the **first round**, I will input the problem and the first step of the solution process.\n"
    "- In **each subsequent round**, I will provide the next step in the solution.\n\n"
    "For each step, you should:\n"
    "- Respond with **\"+\"** if you believe the solution process is correct up to this step.\n"
    "- Respond with **\"-\"** if you detect any issues or errors in the process up to this step.\n\n"
    "Please note:\n"
    "- Only respond with **\"+\"** or **\"-\"**. Do not provide any additional explanations, comments, or justifications.\n\n"
    "Your task is to verify the accuracy and correctness of each step in the given solution process."
)

CONTINUE_PROMPT = "Continue to the next step."


def apply_chat_template(
    tokenizer: AutoTokenizer, messages: List[Dict[str, str]]
) -> str:
    """Apply HF chat template with a safe fallback."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        # Fallback formatting similar to PRMDataset
        parts: List[str] = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        return "\n".join(parts)


@dataclass
class PRMInput:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    reward_positions: List[int]


def build_prm_input(
    tokenizer: AutoTokenizer,
    problem_text: str,
    steps: List[str],
    steps_to_score: int,
    prm_max_length: int,
    pos_token_id: int,
) -> Optional[PRMInput]:
    """
    Construct the PRM conversation and record token indices for '+' evaluations.
    Returns None if the conversation cannot be built within length constraints.
    """
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem_text},
    ]

    reward_positions: List[int] = []
    encoding: Optional[Dict[str, torch.Tensor]] = None

    for step_index, step_text in enumerate(steps):
        if step_index > 0:
            messages.append({"role": "user", "content": CONTINUE_PROMPT})

        content = step_text.strip() or "[No content provided]"
        messages.append({"role": "assistant", "content": content})

        # Update encoding after adding the reasoning step
        conversation_text = apply_chat_template(tokenizer, messages)
        encoding = tokenizer(
            conversation_text,
            return_tensors="pt",
            truncation=True,
            max_length=prm_max_length,
            padding=False,
        )
        if encoding["input_ids"].shape[1] >= prm_max_length:
            return None  # truncated before reaching evaluation tokens

        # Insert an evaluation '+' message for steps to score
        if step_index < steps_to_score:
            messages.append({"role": "assistant", "content": "+"})
            conversation_with_plus = apply_chat_template(tokenizer, messages)
            # print("conversation_with_plus:", conversation_with_plus)
            encoding_plus = tokenizer(
                conversation_with_plus,
                return_tensors="pt",
                truncation=True,
                max_length=prm_max_length,
                padding=False,
            )

            ids_before = encoding["input_ids"][0]
            ids_after = encoding_plus["input_ids"][0]
            len_before = ids_before.size(0)
            new_tokens = ids_after[len_before:]
            plus_positions = [
                len_before + offset
                for offset, token_id in enumerate(new_tokens.tolist())
                if token_id == pos_token_id
            ]
            if not plus_positions:
                return None  # could not locate '+' token cleanly
            reward_positions.extend(plus_positions)
            encoding = encoding_plus  # continue from conversation that includes '+'

    if encoding is None:
        return None

    return PRMInput(
        input_ids=encoding["input_ids"],
        attention_mask=encoding["attention_mask"],
        reward_positions=reward_positions,
    )


def compute_step_probabilities(
    logits: torch.Tensor,
    reward_positions: List[int],
    pos_token_id: int,
    neg_token_id: int,
) -> List[float]:
    """
    Compute P(+) for each evaluation position using a 2-way softmax over +/- logits.
    """
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("Expected logits of shape [1, seq_len, vocab_size]")

    seq_logits = logits[0]  # [seq_len, vocab]
    probs: List[float] = []
    for pos in reward_positions:
        if pos >= seq_logits.shape[0]:
            probs.append(0.0)
            continue
        pos_logit = seq_logits[pos, pos_token_id]
        neg_logit = seq_logits[pos, neg_token_id]
        stacked = torch.stack([pos_logit, neg_logit], dim=0)
        softmax = torch.softmax(stacked.float(), dim=0)
        probs.append(float(softmax[0].item()))
    return probs


# ---------------------------------------------------------------------------
# Lean proof utilities
# ---------------------------------------------------------------------------

LEAN_CODE_BLOCK_RE = re.compile(
    r"```(?:lean4?|lean)\s*(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def extract_lean_code(step_text: str) -> Optional[str]:
    """Extract the first Lean code block from the step text."""
    match = LEAN_CODE_BLOCK_RE.search(step_text or "")
    if match:
        code = match.group(1).strip()
        return code if code else None
    return None


SORRY_PATTERNS = [
    r":=\s*by\s+sorry",
    r":=\s*by\n\s*sorry",
    r":=\s*by\s*\r?\n\s*sorry",
    r":=\s*by\s*\r?\n\s*--.*\n\s*sorry",
    r":=\s*by\s*\r?\n\s*exact\s+?",
]


def replace_sorry(formal_statement: str, proof_body: str) -> Optional[str]:
    """Replace 'sorry' (or similar placeholders) with the provided proof body."""
    proof_body = proof_body.strip()
    if not proof_body:
        return None

    if proof_body.startswith("by"):
        replacement = f":= {proof_body}"
    else:
        replacement = f":= by\n{proof_body}"

    for pattern in SORRY_PATTERNS:
        if re.search(pattern, formal_statement):
            return re.sub(pattern, replacement, formal_statement, count=1)

    if "sorry" in formal_statement:
        return formal_statement.replace("sorry", proof_body, 1)

    return None


def assemble_formal_proof(formal_statement: str, step5_text: str) -> Optional[str]:
    """
    Attempt to construct compilable Lean code by merging the formal statement
    with the proof from step 5.
    """
    proof_code = extract_lean_code(step5_text)
    if proof_code is None:
        return None

    # If the candidate already emitted a full theorem, use it directly.
    if "theorem" in proof_code or "lemma" in proof_code:
        return proof_code

    merged = replace_sorry(formal_statement, proof_code)
    if merged is not None:
        return merged

    # Fallback: append proof with explicit 'by'
    return f"{formal_statement.rstrip()}\nby\n{proof_code}"


# ---------------------------------------------------------------------------
# Candidate data structures
# ---------------------------------------------------------------------------

@dataclass
class CandidateResult:
    index: int
    response: str
    steps: List[str]
    step_scores: List[float]
    mean_score: Optional[float]
    lean_code: Optional[str]
    assembled_code: Optional[str]
    finish_reason: Optional[str]
    selected: bool = False
    full_reasoning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Ensure JSON friendly floats
        if self.mean_score is not None and math.isnan(self.mean_score):
            data["mean_score"] = None
        return data


@dataclass
class ProblemResult:
    problem_id: str
    formal_statement: str
    informal_prefix: str
    candidates: List[CandidateResult]
    selected_index: Optional[int]
    compile_result: Optional[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "formal_statement": self.formal_statement,
            "informal_prefix": self.informal_prefix,
            "selected_index": self.selected_index,
            "compile_result": self.compile_result,
            "candidates": [c.to_dict() for c in self.candidates],
        }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DreamPRM-guided inference for Goedel-Prover"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(REPO_ROOT, "dataset", "minif2f.jsonl"),
        help="Path to minif2f-style JSONL dataset.",
    )
    parser.add_argument(
        "--prover_model",
        type=str,
        required=True,
        help="Path or HF hub ID of the Goedel-Prover model.",
    )
    parser.add_argument(
        "--prm_model",
        type=str,
        required=True,
        help="Path to the DreamPRM model checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store outputs.",
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=1,
        help="Number of prover candidates to sample per problem.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for the prover.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p for prover sampling.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=40960,
        help="Maximum new tokens for prover generation.",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=40960,
        help="Maximum model length for prover (context window).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of problems to process per prover batch.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM.",
    )
    parser.add_argument(
        "--prm_device",
        type=str,
        default="cuda",
        help="Device for the PRM model.",
    )
    parser.add_argument(
        "--prm_dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Floating point dtype for PRM model.",
    )
    parser.add_argument(
        "--prm_max_length",
        type=int,
        default=40690,
        help="Maximum sequence length for PRM tokenizer.",
    )
    parser.add_argument(
        "--steps_to_score",
        type=int,
        default=4,
        help="Number of reasoning steps to score (final step excluded by default).",
    )
    parser.add_argument(
        "--skip_compile",
        action="store_true",
        help="Skip Lean compilation of the selected candidates.",
    )
    parser.add_argument(
        "--compile_workers",
        type=int,
        default=16,
        help="Number of parallel workers for Lean compilation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def str_to_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    data = load_jsonl(args.dataset)
    if not data:
        raise ValueError(f"No data found in {args.dataset}")
    print(f"Loaded {len(data)} problems from {args.dataset}")

    # ------------------------------------------------------------------
    # Initialize prover model (vLLM)
    # ------------------------------------------------------------------
    print("Loading Goedel prover with vLLM...")
    llm = LLM(
        model=args.prover_model,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        n=args.num_candidates,
    )

    # ------------------------------------------------------------------
    # Stage 1: Goedel-Prover inference for all problems
    # ------------------------------------------------------------------
    all_candidates = []  # List[List[CandidateResult]] per problem
    print("Length of dataset:", len(data))
    for start in tqdm(range(0, len(data), args.batch_size), desc="Prover Inference"):
        batch = data[start : start + args.batch_size]
        prompts = [create_simple_prompt(item["formal_statement"]) for item in batch]
        # print("prompts:", prompts)
        vllm_outputs = llm.generate(prompts, sampling_params)
        # print("vllm_outputs:", vllm_outputs)
        for item, llm_output in zip(batch, vllm_outputs):
            candidates = []
            for cand_idx, out in enumerate(llm_output.outputs):
                raw_text = out.text.strip()
                # print("raw_text:", raw_text)
                steps = parse_simple_steps(raw_text)
                # steps = (steps + [""] * 5)[:5]
                # print("steps:", steps)
                candidate = {
                    "index": cand_idx,
                    "response": raw_text,
                    "steps": steps,
                    "finish_reason": getattr(out, "finish_reason", None),
                }
                candidates.append(candidate)
            all_candidates.append({
                "problem": item,
                "candidates": candidates
            })

    # Deallocate prover model to free memory
    del llm
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Initialize PRM model (after prover inference to save memory)
    # ------------------------------------------------------------------
    prm_dtype = str_to_dtype(args.prm_dtype)
    print(f"Loading Process Reward Model from {args.prm_model}...")
    if args.prm_model.endswith('.pt'):
        # Load base model and apply checkpoint
        base_model = "meta-llama/Llama-3.2-1B"  # Assuming this is the base model
        prm_tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
            use_fast=False,
        )
        prm_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            torch_dtype=prm_dtype,
        ).to(args.prm_device)
        checkpoint = torch.load(args.prm_model, map_location='cpu')
        prm_model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded checkpoint from {args.prm_model} onto base model {base_model} (strict=False)")
    else:
        prm_tokenizer = AutoTokenizer.from_pretrained(
            args.prm_model,
            trust_remote_code=True,
            use_fast=False,
        )
        prm_model = AutoModelForCausalLM.from_pretrained(
            args.prm_model,
            trust_remote_code=True,
            torch_dtype=prm_dtype,
        ).to(args.prm_device)
    prm_model.eval()

    pos_tokens = prm_tokenizer.encode("+", add_special_tokens=False)
    neg_tokens = prm_tokenizer.encode("-", add_special_tokens=False)
    if len(pos_tokens) != 1 or len(neg_tokens) != 1:
        raise ValueError("Tokenizer must map '+' and '-' to single tokens.")
    pos_token_id = pos_tokens[0]
    neg_token_id = neg_tokens[0]

    # ------------------------------------------------------------------
    # Stage 2: PRM scoring and selection
    # ------------------------------------------------------------------
    results: List[ProblemResult] = []
    compile_jobs: List[Dict[str, Any]] = []
    num_steps_to_score = max(0, args.steps_to_score)
    for entry in tqdm(all_candidates, desc="PRM Scoring"):
        item = entry["problem"]
        candidates_raw = entry["candidates"]
        problem_id = item.get("problem_id") or item.get("name") or "unknown_problem"
        informal_prefix = item.get("informal_prefix", "")
        formal_statement = item.get("formal_statement", "")
        problem_text = f"{informal_prefix}{formal_statement}"
        candidates: List[CandidateResult] = []
        for cand in candidates_raw:
            steps = cand["steps"]
            steps_available = len([s for s in steps if s.strip()])
            effective_steps_to_score = max(0, steps_available - 1)  # Evaluate all intermediate steps (exclude final Lean step)
            step_scores: List[float] = []
            mean_score: Optional[float] = None
            if effective_steps_to_score > 0:
                prm_input = build_prm_input(
                    tokenizer=prm_tokenizer,
                    problem_text=problem_text,
                    steps=steps,
                    steps_to_score=effective_steps_to_score,
                    prm_max_length=args.prm_max_length,
                    pos_token_id=pos_token_id,
                )
                if (
                    prm_input is not None
                    and len(prm_input.reward_positions) >= effective_steps_to_score
                ):
                    with torch.inference_mode():
                        outputs = prm_model(
                            input_ids=prm_input.input_ids.to(args.prm_device),
                            attention_mask=prm_input.attention_mask.to(args.prm_device),
                        )
                    step_scores = compute_step_probabilities(
                        logits=outputs.logits,
                        reward_positions=prm_input.reward_positions[:effective_steps_to_score],
                        pos_token_id=pos_token_id,
                        neg_token_id=neg_token_id,
                    )
                    if step_scores:
                        mean_score = sum(step_scores) / len(step_scores)
            lean_code = assemble_formal_proof(formal_statement, steps[4] if len(steps) >= 5 else "")
            candidate_obj = CandidateResult(
                index=cand["index"],
                response=cand["response"],
                steps=steps,
                step_scores=[float(x) for x in step_scores],
                mean_score=float(mean_score) if mean_score is not None else None,
                lean_code=lean_code,
                assembled_code=lean_code,
                finish_reason=cand["finish_reason"],
                full_reasoning="\n".join([s for s in steps if s.strip()]),
            )
            candidates.append(candidate_obj)
        # Select best candidate
        selected_index: Optional[int] = None
        if candidates:
            def candidate_score(c: CandidateResult) -> float:
                if c.mean_score is None:
                    return float("-inf")
                if math.isnan(c.mean_score):
                    return float("-inf")
                return c.mean_score
            selected_index = max(range(len(candidates)), key=lambda i: candidate_score(candidates[i]))
            candidates[selected_index].selected = True
        selected_candidate = candidates[selected_index] if selected_index is not None else None
        compile_result: Optional[Dict[str, Any]] = None
        if (
            not args.skip_compile
            and selected_candidate is not None
            and selected_candidate.assembled_code
        ):
            compile_jobs.append(
                {
                    "name": problem_id,
                    "code": selected_candidate.assembled_code,
                    "problem_id": problem_id,
                }
            )
        problem_result = ProblemResult(
            problem_id=problem_id,
            formal_statement=formal_statement,
            informal_prefix=informal_prefix,
            candidates=candidates,
            selected_index=selected_index,
            compile_result=compile_result,
        )
        results.append(problem_result)

    # ------------------------------------------------------------------
    # Compile selected candidates (if requested)
    # ------------------------------------------------------------------
    if not args.skip_compile and compile_jobs:
        print(f"Running Lean compilation for {len(compile_jobs)} selected candidates...")
        compile_outputs = scheduler(compile_jobs, num_workers=args.compile_workers)
        compile_lookup = {entry["name"]: entry for entry in compile_outputs}

        for result in results:
            entry = compile_lookup.get(result.problem_id)
            if entry is not None:
                result.compile_result = entry

    # ------------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------------
    output_path = os.path.join(args.output_dir, "dreamprm_results.json")
    payload = {
        "config": vars(args),
        "results": [res.to_dict() for res in results],
    }
    save_json(output_path, payload)
    print(f"Saved results to {output_path}")

    # Summary
    total = len(results)
    compiled = sum(1 for r in results if r.compile_result is not None)
    passed = sum(
        1
        for r in results
        if r.compile_result
        and r.compile_result.get("compilation_result", {}).get("pass")
    )
    print(f"Total problems: {total}")
    if not args.skip_compile:
        print(f"Compiled: {compiled}")
        print(f"Pass count: {passed}")


if __name__ == "__main__":
    main()
