import json
import os
import sys
import argparse
import re
from collections import defaultdict

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from lean_compiler.repl_scheduler import scheduler

def extract_jobs_from_jsonl(jsonl_path):
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

def parse_name(name):
    m = re.match(r"^(.*)_(mc|cot)_step(\d+)(?:_sample(\d+))?$", name)
    if m:
        problem = m.group(1)
        typ = m.group(2)
        step = int(m.group(3))
        sample = int(m.group(4)) if m.group(4) is not None else None
        return problem, typ, step, sample
    else:
        m = re.match(r"^(.*)_(mc|cot)_step(\d+)", name)
        if m:
            problem = m.group(1)
            typ = m.group(2)
            step = int(m.group(3))
            return problem, typ, step, None
        else:
            parts = name.split("_")
            return parts[0], None, None, None

def summarize_results(results):
    summary = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"pass": 0, "fail": 0})))
    for entry in results:
        name = entry.get("name")
        if not name or "compilation_result" not in entry:
            continue
        complete = entry["compilation_result"].get("complete", False)
        problem, typ, step, sample = parse_name(name)
        if typ is None or step is None:
            continue
        if complete:
            summary[problem][typ][step]["pass"] += 1
        else:
            summary[problem][typ][step]["fail"] += 1
    # Convert defaultdict to dict for JSON serialization
    def dictify(d):
        if isinstance(d, defaultdict):
            d = {k: dictify(v) for k, v in d.items()}
        return d
    return dictify(summary)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonl', type=str, help='Input .jsonl file with completions')
    parser.add_argument('--output_json', type=str, help='Output .json file for compilation results')
    parser.add_argument('--cpu', type=int, default=64, help='Number of parallel workers')
    parser.add_argument('--compiled_json', type=str, help='Compiled output JSON to summarize')
    parser.add_argument('--summary_json', type=str, help='Summary output JSON file')
    args = parser.parse_args()

    if args.input_jsonl and args.output_json:
        jobs = extract_jobs_from_jsonl(args.input_jsonl)
        print(f"Prepared {len(jobs)} compilation jobs.")
        import random
        random.shuffle(jobs)
        outputs = scheduler(jobs, num_workers=args.cpu)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(outputs, f, indent=2)
        print(f"Saved compilation results to {args.output_json}")

    # Summarize results if requested
    if args.compiled_json and args.summary_json:
        with open(args.compiled_json, 'r', encoding='utf-8') as f:
            results = json.load(f)
        summary = summarize_results(results)
        with open(args.summary_json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {args.summary_json}")

if __name__ == '__main__':
    main()