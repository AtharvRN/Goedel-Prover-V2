import json
import os
import sys
import argparse
import re
from collections import defaultdict
from typing import Any, Dict, List, Iterable

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from lean_compiler.repl_scheduler import scheduler

CODE_BLOCK_REGEX = re.compile(
    r'```(?:lean4|lean)\s*\n(.*?)```',
    re.DOTALL | re.IGNORECASE
)


def extract_lean4_blocks(text: str) -> List[str]:
    blocks = CODE_BLOCK_REGEX.findall(text or '')
    cleaned = [block.strip() for block in blocks if block.strip()]
    if cleaned:
        return cleaned

    stripped = (text or '').strip()
    if stripped and any(keyword in stripped for keyword in ('theorem', 'lemma', 'def', 'by')):
        return [stripped]
    return []


def gather_lean_blocks(source: Any) -> List[str]:
    """Normalize different representations of Lean code into a list of blocks."""
    if isinstance(source, dict):
        candidate = source.get('lean4_code')
        blocks: List[str] = []
        if isinstance(candidate, list):
            blocks = [c.strip() for c in candidate if isinstance(c, str) and c.strip()]
        elif isinstance(candidate, str) and candidate.strip():
            blocks = [candidate.strip()]
        if blocks:
            return blocks

        for key in ('content', 'raw_output', 'text', 'code'):
            text_val = source.get(key)
            if isinstance(text_val, str) and text_val.strip():
                blocks = extract_lean4_blocks(text_val)
                if blocks:
                    return blocks
        return []

    if isinstance(source, str):
        return extract_lean4_blocks(source)

    text_val = str(source or '').strip()
    if not text_val:
        return []
    return extract_lean4_blocks(text_val)


def normalize_step_items(obj: Any) -> Iterable[tuple]:
    if isinstance(obj, dict):
        keyed = []
        for key, value in obj.items():
            if isinstance(key, str) and key.startswith('step_'):
                digits = re.sub(r'\D', '', key)
                step_num = int(digits) if digits else 0
                keyed.append((step_num, value))
        for step_num, value in sorted(keyed, key=lambda kv: kv[0]):
            yield step_num, value
    elif isinstance(obj, list):
        for idx, value in enumerate(obj, start=1):
            yield idx, value


def extract_jobs_from_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            problem_id = item.get('problem_id') or item.get('name')
            name = item.get('name', problem_id)
            sample_id = item.get('sample_id')

            base_name = name or 'item'
            if sample_id is not None:
                base_name = f'{base_name}_sample{sample_id}'

            mc = item.get('mc_proof_completions', {})
            for step_num, samples in normalize_step_items(mc):
                if not isinstance(samples, list):
                    continue
                for sample_idx, sample in enumerate(samples):
                    blocks = gather_lean_blocks(sample)
                    if not blocks:
                        continue
                    for block_idx, code in enumerate(blocks):
                        job_name = f'{base_name}_mc_step{step_num}_sample{sample_idx}'
                        if len(blocks) > 1:
                            job_name += f'_block{block_idx}'
                        jobs.append({
                            'name': job_name,
                            'code': code,
                            'problem_id': problem_id,
                            'step': step_num,
                            'sample_idx': sample_idx,
                            'block_idx': block_idx if len(blocks) > 1 else None,
                            'source': 'mc'
                        })

            cot_steps = item.get('cot_steps', [])
            for step_num, step in normalize_step_items(cot_steps):
                blocks = gather_lean_blocks(step)
                if not blocks:
                    continue
                for block_idx, code in enumerate(blocks):
                    job_name = f'{base_name}_cot_step{step_num}'
                    if len(blocks) > 1:
                        job_name += f'_block{block_idx}'
                    jobs.append({
                        'name': job_name,
                        'code': code,
                        'problem_id': problem_id,
                        'step': step_num,
                        'sample_idx': None,
                        'block_idx': block_idx if len(blocks) > 1 else None,
                        'source': 'cot'
                    })

    return jobs

def parse_name(name):
    m = re.match(r"^(.*)_(mc|cot)_step(\d+)(?:_sample(\d+))?(?:_block(\d+))?$", name)
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
