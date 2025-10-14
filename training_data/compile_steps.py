#!/usr/bin/env python3
"""
Lean 4 compilation script for variable-length CoT with MC completions.

Extracts Lean4 code from:
1. All CoT steps (variable number)
2. MC proof completions (variable configurations)

Compiles all extracted code in parallel.
"""

import json
import os
import sys
import argparse
import re
from typing import List, Dict, Any

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from lean_compiler.repl_scheduler import scheduler


def extract_lean4_code(text: str) -> List[str]:
    """Extract all Lean4 code blocks from text.
    
    Returns a list of code strings (without backticks).
    If no code blocks found, returns the original text as a single-item list.
    """
    # Match code blocks with lean4 or lean language identifier
    code_block_regex = re.compile(
        r'```(?:lean4|lean)\s*\n(.*?)```',
        re.DOTALL | re.IGNORECASE
    )
    matches = code_block_regex.findall(text)
    
    if matches:
        return [match.strip() for match in matches if match.strip()]
    
    # If no code blocks found but text looks like Lean code, return it
    text = text.strip()
    if text and ('theorem' in text or 'lemma' in text or 'def' in text or 'by' in text):
        return [text]
    
    return []


def extract_jobs_from_jsonl(
    jsonl_path: str,
    include_cot_steps: bool = True,
    include_mc_completions: bool = True,
    only_first_lean_block: bool = False
) -> List[Dict[str, Any]]:
    """Extract all Lean4 compilation jobs from JSONL.
    
    Args:
        jsonl_path: Path to input JSONL file
        include_cot_steps: Include Lean4 code from CoT steps
        include_mc_completions: Include Lean4 code from MC completions
        only_first_lean_block: If True, only use first Lean4 block per source
    
    Returns:
        List of job dictionaries for compilation
    """
    def gather_lean_blocks(source: Any) -> List[str]:
        """Normalize different source formats into a list of Lean blocks."""
        blocks: List[str] = []

        if isinstance(source, dict):
            candidate = source.get('lean4_code')
            if isinstance(candidate, list):
                blocks = [c.strip() for c in candidate if isinstance(c, str) and c.strip()]
            elif isinstance(candidate, str):
                blocks = [candidate.strip()] if candidate.strip() else []

            if not blocks:
                # Fallback to raw textual fields
                for key in ('content', 'raw_output', 'text', 'code'):
                    text_val = source.get(key)
                    if isinstance(text_val, str) and text_val.strip():
                        blocks = extract_lean4_code(text_val)
                        if blocks:
                            break
        elif isinstance(source, str):
            blocks = extract_lean4_code(source)
        else:
            text_val = str(source or '').strip()
            if text_val:
                blocks = extract_lean4_code(text_val)

        return [b for b in blocks if b.strip()]

    jobs = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON at line {line_num}: {e}")
                continue
            
            problem_id = item.get('problem_id') or item.get('name', f'item_{line_num}')
            sample_id = item.get('sample_id')
            
            # Construct base name with sample info if present
            base_name = problem_id
            if sample_id is not None:
                base_name = f"{problem_id}_sample{sample_id}"
            
            # Extract from CoT steps
            if include_cot_steps:
                cot_steps = item.get('cot_steps', [])
                
                # Handle both list format (new) and dict format (old)
                if isinstance(cot_steps, list):
                    # New format: list of dicts or legacy strings
                    for idx, step in enumerate(cot_steps, start=1):
                        step_num = idx
                        extracted: List[str] = []

                        if isinstance(step, dict):
                            step_num = step.get('step_num', idx)
                            extracted = gather_lean_blocks(step)
                        else:
                            extracted = gather_lean_blocks(step)

                        if not extracted:
                            continue

                        codes_to_use = extracted[:1] if only_first_lean_block else extracted
                        for block_idx, code in enumerate(codes_to_use):
                            job_name = f"{base_name}_cot_step{step_num}"
                            if len(codes_to_use) > 1:
                                job_name += f"_block{block_idx}"
                            
                            jobs.append({
                                'name': job_name,
                                'code': code,
                                'problem_id': problem_id,
                                'sample_id': sample_id,
                                'step': step_num,
                                'block_idx': block_idx if len(codes_to_use) > 1 else None,
                                'source': 'cot_step'
                            })
                
                elif isinstance(cot_steps, dict):
                    # Old format: dict with step_1, step_2, etc.
                    step_items = sorted(
                        ((key, value) for key, value in cot_steps.items() if key.startswith('step_')),
                        key=lambda kv: int(re.sub(r'\D', '', kv[0]) or 0)
                    )

                    for step_key, step_content in step_items:
                        step_num = int(re.sub(r'\D', '', step_key) or 0)
                        extracted = gather_lean_blocks(step_content)
                        if not extracted:
                            continue

                        codes_to_use = extracted[:1] if only_first_lean_block else extracted
                        for block_idx, code in enumerate(codes_to_use):
                            job_name = f"{base_name}_cot_step{step_num}"
                            if len(codes_to_use) > 1:
                                job_name += f"_block{block_idx}"
                            
                            jobs.append({
                                'name': job_name,
                                'code': code,
                                'problem_id': problem_id,
                                'sample_id': sample_id,
                                'step': step_num,
                                'block_idx': block_idx if len(codes_to_use) > 1 else None,
                                'source': 'cot_step'
                            })
            
            # Extract from MC completions
            if include_mc_completions:
                mc = item.get('mc_proof_completions', [])
                
                # Handle both list format (new) and dict format (old)
                if isinstance(mc, list):
                    # New format: list of configuration dicts
                    for config in mc:
                        if not isinstance(config, dict):
                            continue
                        
                        config_idx = config.get('config_index', 0)
                        num_steps = config.get('num_steps_used', 0)
                        completions = config.get('completions', [])
                        
                        for comp_idx, completion in enumerate(completions):
                            extracted = gather_lean_blocks(completion)
                            codes_to_use = extracted[:1] if only_first_lean_block else extracted
                            for block_idx, code in enumerate(codes_to_use):
                                job_name = f"{base_name}_mc_config{config_idx}_steps{num_steps}_sample{comp_idx}"
                                if len(codes_to_use) > 1:
                                    job_name += f"_block{block_idx}"
                                
                                jobs.append({
                                    'name': job_name,
                                    'code': code,
                                    'problem_id': problem_id,
                                    'sample_id': sample_id,
                                    'config_index': config_idx,
                                    'num_steps_used': num_steps,
                                    'completion_idx': comp_idx,
                                    'block_idx': block_idx if len(codes_to_use) > 1 else None,
                                    'source': 'mc_completion'
                                })
                
                elif isinstance(mc, dict):
                    # Old format: dict with step_1, step_2, etc.
                    for step in range(1, 100):  # Support up to 100 steps
                        step_key = f'step_{step}'
                        if step_key not in mc:
                            break
                        
                        samples = mc[step_key]
                        if not isinstance(samples, list):
                            continue
                        
                        for sample_idx, sample in enumerate(samples):
                            extracted = gather_lean_blocks(sample)
                            if not extracted:
                                continue

                            codes_to_use = extracted[:1] if only_first_lean_block else extracted
                            for block_idx, lean_code in enumerate(codes_to_use):
                                job_name = f"{base_name}_mc_step{step}_sample{sample_idx}"
                                if len(codes_to_use) > 1:
                                    job_name += f"_block{block_idx}"
                                
                                jobs.append({
                                    'name': job_name,
                                    'code': lean_code,
                                    'problem_id': problem_id,
                                    'sample_id': sample_id,
                                    'step': step,
                                    'completion_idx': sample_idx,
                                    'block_idx': block_idx if len(codes_to_use) > 1 else None,
                                    'source': 'mc_completion'
                                })
    
    return jobs


def main():
    parser = argparse.ArgumentParser(
        description='Compile Lean4 code from CoT and MC completions'
    )
    parser.add_argument('--input_jsonl', type=str, required=True, 
                       help='Input .jsonl file with completions')
    parser.add_argument('--output_json', type=str, required=True, 
                       help='Output .json file for compilation results')
    parser.add_argument('--cpu', type=int, default=64, 
                       help='Number of parallel workers')
    parser.add_argument('--no_cot', action='store_true',
                       help='Skip CoT steps (only compile MC completions)')
    parser.add_argument('--no_mc', action='store_true',
                       help='Skip MC completions (only compile CoT steps)')
    parser.add_argument('--only_first_block', action='store_true',
                       help='Only compile first Lean4 block per source')
    parser.add_argument('--shuffle', action='store_true',
                       help='Shuffle jobs before compilation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for shuffling')
    
    args = parser.parse_args()

    print(f"Extracting Lean4 code from: {args.input_jsonl}")
    jobs = extract_jobs_from_jsonl(
        args.input_jsonl,
        include_cot_steps=not args.no_cot,
        include_mc_completions=not args.no_mc,
        only_first_lean_block=args.only_first_block
    )
    
    print(f"Prepared {len(jobs)} compilation jobs.")
    
    if len(jobs) == 0:
        print("Warning: No jobs extracted. Check input file format.")
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2)
        return
    
    # Print summary
    cot_jobs = sum(1 for j in jobs if j['source'] == 'cot_step')
    mc_jobs = sum(1 for j in jobs if j['source'] == 'mc_completion')
    print(f"  - CoT steps: {cot_jobs}")
    print(f"  - MC completions: {mc_jobs}")

    if args.shuffle:
        import random
        random.seed(args.seed)
        random.shuffle(jobs)
        print(f"Shuffled jobs with seed={args.seed}")

    print(f"Starting compilation with {args.cpu} workers...")
    outputs = scheduler(jobs, num_workers=args.cpu)

    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, indent=2)
    
    # Print success statistics
    success_count = sum(1 for o in outputs if o.get('success', False))
    success_rate = (success_count / len(outputs) * 100) if outputs else 0
    print(f"\nCompilation complete:")
    print(f"  - Total jobs: {len(outputs)}")
    print(f"  - Successful: {success_count} ({success_rate:.1f}%)")
    print(f"  - Failed: {len(outputs) - success_count}")
    print(f"\nSaved compilation results to {args.output_json}")


if __name__ == '__main__':
    main()
