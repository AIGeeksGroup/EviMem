#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IRR Runner V5 - LoCoMo evaluation runner for IRRv5.

V5 improvements: deeply optimized version
- Entity tracking
- "Not mentioned" detection
- Enhanced temporal reasoning (INFERRABLE)
- Multi-hop reasoning chains
- Dual-path retrieval
- Answer certainty optimization

Usage (OpenAI):
    python irr_runner_v5.py --db "path/to/conv26.sqlite.db" \
                            --api-key "sk-xxx" \
                            --locomo \
                            --indices conv-26 \
                            --category 2 \
                            --model gpt-4o \
                            --provider openai

Usage (Gemini):
    python irr_runner_v5.py --db "path/to/conv26.sqlite.db" \
                            --api-key "your-gemini-key" \
                            --locomo \
                            --indices conv-26 \
                            --category 2 \
                            --model gemini-2.0-flash \
                            --provider google_ai

New arguments:
    --temporal-inference     allow temporal inference from clues (default True)
    --no-temporal-inference  disable temporal inference
    --entity-tracking        enable entity tracking (default True)
    --no-entity-tracking     disable entity tracking
    --dual-retrieval         enable dual-path retrieval (default True)
    --no-dual-retrieval      disable dual-path retrieval
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# Add EviMem repo root to sys.path so `src.X.Y` imports work from any cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.memory_retrieval_system.IRIS_retriever_v5 import create_irr_retriever_v5, IRRResult


def setup_api_key(api_key: str, provider: str = "openai"):
    """Configure the API key."""
    if provider == "openai":
        os.environ["OPENAI_API_KEY"] = api_key
    elif provider == "google_ai":
        os.environ["GEMINI_API_KEY"] = api_key
    print(f"[Setup] {provider.upper()} API key configured (length: {len(api_key)})")


def load_locomo_data(locomo_path: str = "locomo/data/locomo10.json") -> List[Dict]:
    """Load the LoCoMo dataset."""
    with open(locomo_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_sample_index(data: List[Dict], sample_id: str) -> int:
    """Find the index in the dataset matching sample_id."""
    for i, item in enumerate(data):
        if item.get('sample_id') == sample_id:
            return i
    raise ValueError(f"Sample ID '{sample_id}' not found in dataset")


def select_questions(
    case_data: Dict,
    category: Optional[int] = None,
    num_questions: int = 9999,
    offset: int = 0
) -> List[Dict]:
    """Filter questions."""
    all_qa = case_data.get('qa', [])

    if category is not None:
        all_qa = [q for q in all_qa if q.get('category') == category]

    return all_qa[offset:offset + num_questions]


def run_irr_v5_evaluation(
    db_path: str,
    api_key: str,
    locomo_path: str,
    sample_ids: List[str],
    category: Optional[int] = None,
    num_questions: int = 9999,
    question_offset: int = 0,
    model: str = "gpt-4o",
    provider: str = "openai",
    max_iterations: int = 3,
    allow_approximate: bool = True,
    temporal_allow_inference: bool = True,
    enable_entity_tracking: bool = True,
    enable_dual_retrieval: bool = True,
    output_dir: str = "results/memory_retrieval/irr_v5"
):
    """
    Run IRRv5 evaluation.

    Args:
        db_path: database path (supports {sample_id} placeholder)
        api_key: API key
        locomo_path: path to LoCoMo dataset
        sample_ids: list of sample_ids to evaluate
        category: filter questions by category
        num_questions: questions per case
        question_offset: question offset
        model: model used for answer generation
        provider: LLM provider
        max_iterations: maximum number of iterations
        allow_approximate: whether to allow approximate reasoning
        temporal_allow_inference: whether temporal questions allow inference
        enable_entity_tracking: whether to enable entity tracking
        enable_dual_retrieval: whether to enable dual-path retrieval
        output_dir: output directory
    """
    setup_api_key(api_key, provider)

    # Load LoCoMo data
    locomo_data = load_locomo_data(locomo_path)
    print(f"[Data] Loaded {len(locomo_data)} cases from LoCoMo")

    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for sample_id in sample_ids:
        print(f"\n{'='*60}")
        print(f"Processing: {sample_id}")
        print(f"{'='*60}")

        # Find the matching case
        try:
            idx = find_sample_index(locomo_data, sample_id)
            case = locomo_data[idx]
        except ValueError as e:
            print(f"[Error] {e}")
            continue

        # Resolve database path
        actual_db_path = db_path.replace("{sample_id}", sample_id)
        if not Path(actual_db_path).exists():
            print(f"[Error] Database not found: {actual_db_path}")
            continue

        # Create IRRv5 retriever
        print(f"[Init] Creating IRRv5 Retriever with db: {actual_db_path}")
        print(f"[Config] allow_approximate={allow_approximate}, temporal_inference={temporal_allow_inference}")
        print(f"[Config] entity_tracking={enable_entity_tracking}, dual_retrieval={enable_dual_retrieval}")

        irr = create_irr_retriever_v5(
            db_path=actual_db_path,
            openai_api_key=api_key,
            provider=provider,
            model=model,
            max_iterations=max_iterations,
            allow_approximate_reasoning=allow_approximate,
            temporal_allow_inference=temporal_allow_inference,
            enable_entity_tracking=enable_entity_tracking,
            enable_dual_retrieval=enable_dual_retrieval,
            verbose=True
        )

        # Filter questions
        questions = select_questions(case, category, num_questions, question_offset)
        if not questions:
            print(f"[Warning] No questions selected for {sample_id}")
            continue

        print(f"[Questions] Selected {len(questions)} questions (Category: {category if category else 'All'})")

        # Run evaluation
        results = []
        start_time = time.time()

        for i, qa in enumerate(questions, 1):
            question = qa['question']
            expected = qa.get('answer') or qa.get('adversarial_answer', 'Unknown')
            cat = qa.get('category', 0)

            print(f"\n[Q{i}/{len(questions)}] Category {cat}")
            print(f"Question: {question}")
            print(f"Expected: {expected}")

            try:
                irr_result: IRRResult = irr.retrieve_with_irr(
                    question=question,
                    base_top_k=10,
                    expand_hops=1,
                    model=model
                )

                result_dict = {
                    "question_index": i - 1,
                    "question": question,
                    "expected_answer": expected,
                    "generated_answer": irr_result.final_answer,
                    "category": cat,
                    "confidence": irr_result.confidence,
                    "is_approximate": irr_result.is_approximate,
                    "is_not_mentioned": irr_result.is_not_mentioned,
                    "is_temporal_question": irr_result.is_temporal_question,
                    "entities_tracked": irr_result.entities_tracked,
                    "has_reasoning_chain": irr_result.reasoning_chain is not None,
                    "reasoning_steps": len(irr_result.reasoning_chain.steps) if irr_result.reasoning_chain else 0,
                    "iterations": len(irr_result.iterations),
                    "total_facts": irr_result.total_facts,
                    "retrieval_time": irr_result.retrieval_time,
                    "generation_time": irr_result.generation_time,
                    "total_time": irr_result.total_time,
                    "iteration_details": [
                        {
                            "iteration": it.iteration,
                            "query_original": it.query_original,
                            "query_refined": it.query_refined,
                            "facts_from_original": it.facts_from_original,
                            "facts_from_refined": it.facts_from_refined,
                            "facts_expanded": it.facts_expanded,
                            "facts_count": len(it.facts_from_original) + len(it.facts_from_refined),
                            "has_exact": it.has_exact_match,
                            "has_inferrable": it.has_inferrable_info,
                            "has_partial": it.has_partial_match,
                            "confidence": it.confidence,
                            "missing_info": it.missing_info,
                            "sufficiency_prompt": it.sufficiency_prompt,
                            "sufficiency_response": it.sufficiency_response,
                            "refine_prompt": it.refine_prompt,
                            "refine_response": it.refine_response
                        }
                        for it in irr_result.iterations
                    ],
                    "reasoning_chain": {
                        "steps": irr_result.reasoning_chain.steps,
                        "entities_involved": irr_result.reasoning_chain.entities_involved,
                        "confidence": irr_result.reasoning_chain.confidence
                    } if irr_result.reasoning_chain else None,
                    "all_facts": list(dict.fromkeys(
                        sum([it.facts_from_original + it.facts_from_refined + it.facts_expanded for it in irr_result.iterations], [])
                    ))
                }

                results.append(result_dict)

                print(f"Generated: {irr_result.final_answer[:150]}")
                print(f"Confidence: {irr_result.confidence:.2f}")
                print(f"Flags: Approximate={irr_result.is_approximate}, NotMentioned={irr_result.is_not_mentioned}")
                if irr_result.entities_tracked:
                    print(f"Entities: {', '.join(irr_result.entities_tracked)}")
                if irr_result.reasoning_chain:
                    print(f"Reasoning: {len(irr_result.reasoning_chain.steps)} steps")

            except Exception as e:
                print(f"[Error] Failed to process question: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    "question_index": i - 1,
                    "question": question,
                    "expected_answer": expected,
                    "generated_answer": f"ERROR: {str(e)}",
                    "category": cat,
                    "error": str(e)
                })

        total_time = time.time() - start_time

        # Statistics
        temporal_questions = sum(1 for r in results if r.get("is_temporal_question", False))
        not_mentioned_count = sum(1 for r in results if r.get("is_not_mentioned", False))
        approximate_count = sum(1 for r in results if r.get("is_approximate", False))
        with_entities = sum(1 for r in results if r.get("entities_tracked", []))
        with_reasoning = sum(1 for r in results if r.get("has_reasoning_chain", False))

        # Save results
        cat_suffix = f"_Cat{category}" if category else ""
        offset_suffix = f"_Q{question_offset+1}-{question_offset+len(questions)}" if question_offset > 0 or num_questions < 9999 else ""

        output_file = output_path / f"IRRv5_{sample_id}{offset_suffix}{cat_suffix}_{timestamp}.json"

        output_data = {
            "source_file": str(output_file),
            "metadata": {
                "sample_id": sample_id,
                "category": category,
                "question_offset": question_offset,
                "questions_count": len(questions),
                "model": model,
                "provider": provider,
                "max_iterations": max_iterations,
                "allow_approximate_reasoning": allow_approximate,
                "temporal_allow_inference": temporal_allow_inference,
                "enable_entity_tracking": enable_entity_tracking,
                "enable_dual_retrieval": enable_dual_retrieval,
                "architecture": "irr_v5",
                "timestamp": timestamp
            },
            "summary": {
                "total_questions": len(results),
                "temporal_questions": temporal_questions,
                "not_mentioned_responses": not_mentioned_count,
                "approximate_responses": approximate_count,
                "questions_with_entities": with_entities,
                "questions_with_reasoning_chains": with_reasoning,
                "total_time_seconds": total_time,
                "avg_time_per_question": total_time / len(results) if results else 0
            },
            "results": results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print(f"[Completed] {sample_id}")
        print(f"Total questions: {len(results)}")
        print(f"Temporal questions: {temporal_questions}")
        print(f"'Not mentioned' responses: {not_mentioned_count}")
        print(f"Approximate responses: {approximate_count}")
        print(f"Questions with entities tracked: {with_entities}")
        print(f"Questions with reasoning chains: {with_reasoning}")
        print(f"Total time: {total_time:.2f}s (Avg: {total_time/len(results):.2f}s/q)")
        print(f"Results saved to: {output_file}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="IRR V5 Runner for LoCoMo evaluation")

    # Database and dataset
    parser.add_argument("--db", type=str, required=True,
                        help="Database path (use {sample_id} as placeholder)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenAI API key or Gemini API key. "
                             "If omitted, falls back to OPENAI_API_KEY / "
                             "GEMINI_API_KEY environment variable.")
    parser.add_argument("--locomo", action="store_true",
                        help="Use LoCoMo dataset")
    parser.add_argument("--locomo-path", type=str, default="locomo/data/locomo10.json",
                        help="Path to LoCoMo dataset")

    # Sample and question selection
    parser.add_argument("--indices", type=str, required=True,
                        help="Sample IDs to evaluate (e.g., 'conv-26,conv-30' or 'conv-26')")
    parser.add_argument("--category", type=int, default=None,
                        help="Filter questions by category (1-5)")
    parser.add_argument("--questions-per-case", type=int, default=9999,
                        help="Number of questions per case")
    parser.add_argument("--question-offset", type=int, default=0,
                        help="Question offset (skip first N questions)")

    # Model configuration
    parser.add_argument("--provider", type=str, default="openai",
                        choices=["openai", "google_ai"],
                        help="LLM provider")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="Model to use")

    # IRR configuration
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="Maximum IRR iterations")
    parser.add_argument("--approximate", action="store_true", default=True,
                        help="Allow approximate reasoning (default: True)")
    parser.add_argument("--no-approximate", dest="approximate", action="store_false",
                        help="Disable approximate reasoning")

    # V5 new configuration
    parser.add_argument("--temporal-inference", action="store_true", default=True,
                        help="Allow temporal inference from clues (default: True)")
    parser.add_argument("--no-temporal-inference", dest="temporal_inference", action="store_false",
                        help="Disable temporal inference")

    parser.add_argument("--entity-tracking", action="store_true", default=True,
                        help="Enable entity tracking (default: True)")
    parser.add_argument("--no-entity-tracking", dest="entity_tracking", action="store_false",
                        help="Disable entity tracking")

    parser.add_argument("--dual-retrieval", action="store_true", default=True,
                        help="Enable dual-path retrieval (default: True)")
    parser.add_argument("--no-dual-retrieval", dest="dual_retrieval", action="store_false",
                        help="Disable dual-path retrieval")

    # Output
    parser.add_argument("--output-dir", type=str, default="results/memory_retrieval/irr_v5",
                        help="Output directory")

    args = parser.parse_args()

    # Parse sample_ids
    sample_ids = [s.strip() for s in args.indices.split(",")]

    # Resolve API key: CLI arg > env var
    env_var = "OPENAI_API_KEY" if args.provider == "openai" else "GEMINI_API_KEY"
    api_key = args.api_key or os.environ.get(env_var)
    if not api_key:
        print(f"[Error] No API key provided. Use --api-key or set {env_var}")
        sys.exit(1)

    # Run evaluation
    run_irr_v5_evaluation(
        db_path=args.db,
        api_key=api_key,
        locomo_path=args.locomo_path,
        sample_ids=sample_ids,
        category=args.category,
        num_questions=args.questions_per_case,
        question_offset=args.question_offset,
        model=args.model,
        provider=args.provider,
        max_iterations=args.max_iterations,
        allow_approximate=args.approximate,
        temporal_allow_inference=args.temporal_inference,
        enable_entity_tracking=args.entity_tracking,
        enable_dual_retrieval=args.dual_retrieval,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
