import json
import os
import sys
import argparse

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from lean_compiler.repl_scheduler import scheduler

def extract_jobs_from_jsonl(jsonl_path):
    import re
    def extract_lean4_code(text):
        match = re.search(r"```lean4(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    jobs = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            problem_id = item.get('problem_id') or item.get('name')
            name = item.get('name', problem_id)
            # MC completions: step_1 to step_4, each with 12 samples
            mc = item.get('mc_proof_completions', {})
            for step in range(1, 5):
                samples = mc.get(f'step_{step}', [])
                for idx, code in enumerate(samples):
                    lean_code = extract_lean4_code(code)
                    jobs.append({
                        'name': f'{name}_mc_step{step}_sample{idx}',
                        'code': lean_code,
                        'problem_id': problem_id,
                        'step': step,
                        'sample_idx': idx,
                        'source': 'mc'
                    })
            # CoT step 5: single string
            cot_steps = item.get('cot_steps', {})
            step5 = cot_steps.get('step_5', '') if isinstance(cot_steps, dict) else (cot_steps[4] if isinstance(cot_steps, list) and len(cot_steps) >= 5 else '')
            if step5:
                lean_code = extract_lean4_code(step5)
                jobs.append({
                    'name': f'{name}_cot_step5',
                    'code': lean_code,
                    'problem_id': problem_id,
                    'step': 5,
                    'sample_idx': None,
                    'source': 'cot'
                })
    return jobs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonl', type=str, required=True, help='Input .jsonl file with completions')
    parser.add_argument('--output_json', type=str, required=True, help='Output .json file for compilation results')
    parser.add_argument('--cpu', type=int, default=64, help='Number of parallel workers')
    args = parser.parse_args()

    jobs = extract_jobs_from_jsonl(args.input_jsonl)
    print(f"Prepared {len(jobs)} compilation jobs.")

    import random
    random.shuffle(jobs)

    outputs = scheduler(jobs, num_workers=args.cpu)

    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, indent=2)
    print(f"Saved compilation results to {args.output_json}")

if __name__ == '__main__':
    main()