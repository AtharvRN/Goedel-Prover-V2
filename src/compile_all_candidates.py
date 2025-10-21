#!/usr/bin/env python3
"""
Compile every DreamPRM candidate and report pass@k statistics.

Given a dreamprm_results.json (produced by dreamprm_inference.py), this script:
1. Extracts every candidate Lean proof (if available).
2. Runs the Lean REPL scheduler on all candidates, collecting compile results.
3. Computes pass@k metrics (k=1..num_candidates) based on candidate ordering.
4. Saves a detailed JSON report so downstream analyses can reuse the data.
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from lean_compiler.repl_scheduler import scheduler  # type: ignore


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    if chunk_size <= 0:
        return [items]
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def prepare_jobs(results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Tuple[int, int]]]:
    """Return scheduler jobs and mapping from job name to (problem_idx, candidate_idx)."""
    jobs: List[Dict[str, Any]] = []
    index_lookup: Dict[str, Tuple[int, int]] = {}

    for p_idx, problem in enumerate(results):
        candidates = problem.get("candidates", [])
        for c_idx, cand in enumerate(candidates):
            code = cand.get("assembled_code") or ""
            if not code.strip():
                continue
            job_name = f"{problem.get('problem_id', f'problem_{p_idx}')}_cand{c_idx}"
            jobs.append({
                "name": job_name,
                "code": code,
                "problem_id": problem.get("problem_id"),
            })
            index_lookup[job_name] = (p_idx, c_idx)

    return jobs, index_lookup


def compile_jobs(jobs: List[Dict[str, Any]], num_workers: int) -> List[Dict[str, Any]]:
    """Run the Lean scheduler in batches to avoid overwhelming the REPL."""
    if not jobs:
        return []

    compiled: List[Dict[str, Any]] = []
    # Draining all jobs at once is feasible, but batching keeps memory lower.
    for chunk in chunk_list(jobs, max(1, num_workers * 32)):
        compiled.extend(scheduler(chunk, num_workers=num_workers))
    return compiled


def attach_results(
    results: List[Dict[str, Any]],
    index_lookup: Dict[str, Tuple[int, int]],
    compiled_outputs: List[Dict[str, Any]],
) -> None:
    """Attach compilation outputs back to the candidate entries in-place."""
    for entry in compiled_outputs:
        job_name = entry.get("name")
        if job_name not in index_lookup:
            continue
        p_idx, c_idx = index_lookup[job_name]
        candidate = results[p_idx]["candidates"][c_idx]
        candidate["compile_result"] = entry.get("compilation_result", {})


def compute_pass_at_k(results: List[Dict[str, Any]], max_k: int) -> Dict[int, float]:
    """Compute pass@k statistics for k=1..max_k."""
    totals = {k: 0 for k in range(1, max_k + 1)}
    passes = {k: 0 for k in range(1, max_k + 1)}

    for problem in results:
        candidates = problem.get("candidates", [])
        num_candidates = len(candidates)
        if num_candidates == 0:
            continue

        max_considered = min(max_k, num_candidates)
        pass_prefix = [False] * max_considered
        for k in range(1, max_considered + 1):
            totals[k] += 1
            for cand in candidates[:k]:
                compile_res = cand.get("compile_result") or {}
                if compile_res.get("pass"):
                    pass_prefix[k - 1] = True
                    break
            if pass_prefix[k - 1]:
                passes[k] += 1

    return {
        k: (passes[k] / totals[k] * 100) if totals[k] else math.nan
        for k in range(1, max_k + 1)
    }


def summarise(results: List[Dict[str, Any]], num_candidates: int) -> Dict[str, Any]:
    total_problems = len(results)
    problems_with_code = sum(
        1 for r in results if any((c.get("assembled_code") or "").strip() for c in r.get("candidates", []))
    )
    pass_counts = 0
    compiled_counts = 0

    for problem in results:
        best_compile = False
        candidates = problem.get("candidates", [])
        for cand in candidates:
            compile_res = cand.get("compile_result") or {}
            if compile_res:
                compiled_counts += 1
            if compile_res.get("pass"):
                best_compile = True
        if best_compile:
            pass_counts += 1

    pass_at_k = compute_pass_at_k(results, num_candidates)

    return {
        "total_problems": total_problems,
        "problems_with_any_code": problems_with_code,
        "problems_with_any_pass": pass_counts,
        "total_candidate_compiles": compiled_counts,
        "pass_at_k": pass_at_k,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile every DreamPRM candidate and compute pass@k.")
    parser.add_argument(
        "--input_json",
        type=str,
        default="dreamprm_results.json",
        help="Path to dreamprm_results.json",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="dreamprm_candidate_compilation.json",
        help="Path to store the enriched results",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of parallel Lean workers",
    )
    args = parser.parse_args()

    data = load_json(args.input_json)
    results = data.get("results", [])
    if not results:
        print("No results found in input JSON.", file=sys.stderr)
        return

    num_candidates = max(len(r.get("candidates", [])) for r in results) if results else 0

    jobs, index_lookup = prepare_jobs(results)
    print(f"Prepared {len(jobs)} candidate jobs across {len(results)} problems.")

    compiled_outputs = compile_jobs(jobs, num_workers=args.num_workers)
    print(f"Compilation finished for {len(compiled_outputs)} jobs.")

    attach_results(results, index_lookup, compiled_outputs)

    summary = summarise(results, num_candidates)

    output_payload = {
        "config": data.get("config", {}),
        "summary": summary,
        "results": results,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    print("Summary:")
    for key, value in summary.items():
        if key == "pass_at_k":
            for k, pct in value.items():
                print(f"  pass@{k}: {pct:.2f}%")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
