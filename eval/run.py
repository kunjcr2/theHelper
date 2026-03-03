"""
eval/run.py — Evaluation runner.

Loads a dataset JSON, runs each question through the RAG pipeline,
computes metrics, and saves reports to eval/reports/<run_id>.[json|md].

Dataset format:
[
  {
    "question": "...",
    "doc_id": "...",          // used to check if right doc was retrieved
    "expected_sources": ["chunk_id_or_doc_id", ...],  // optional
    "ground_truth": "..."     // optional, for future reference
  }
]

Returns exit code 0 if all thresholds pass, 1 otherwise.
"""

import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Quality thresholds (CI gate) ──────────────────────────────────────────────
HIT_RATE_THRESHOLD = 0.5   # hit-rate@k must be >= this to pass
RECALL_THRESHOLD = 0.4     # recall@k must be >= this to pass


def run_eval(
    dataset_path: Path,
    output_dir: Optional[Path] = None,
    k: int = 5,
) -> int:
    """
    Run evaluation and write reports.
    Returns 0 on pass, 1 on threshold failure.
    """
    from rag.pipeline import RAGPipeline
    from rag.config import load_config
    from eval.metrics import hit_rate_at_k, recall_at_k, faithfulness_score, relevance_score

    output_dir = output_dir or Path("eval/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    config = load_config()
    pipeline = RAGPipeline(config)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    results = []

    api_key = os.getenv("OPENAI_API_KEY")
    openai_client = None
    if api_key:
        from openai import OpenAI
        openai_client = OpenAI(api_key=api_key)

    for i, item in enumerate(dataset):
        question = item["question"]
        expected_sources = item.get("expected_sources", [])
        ground_truth = item.get("ground_truth", "")

        logger.info("[%d/%d] %s", i + 1, len(dataset), question[:80])

        try:
            qr = pipeline.query(question, k=k)
            retrieved_ids = [c.chunk_id for c in qr.retrieved_chunks]
            # Also include doc_ids for broader matching
            retrieved_doc_ids = list({c.doc_id for c in qr.retrieved_chunks})

            # Use chunk_ids OR doc_ids for hit-rate (expected_sources can be either)
            all_retrieved = retrieved_ids + retrieved_doc_ids
            hr = hit_rate_at_k(all_retrieved, expected_sources, k)
            rc = recall_at_k(all_retrieved, expected_sources, k)

            context_text = "\n\n".join(c.text for c in qr.retrieved_chunks)
            faith = 0.5
            relev = 0.5
            if openai_client and qr.answer and "failed" not in qr.answer.lower():
                faith = faithfulness_score(qr.answer, context_text, openai_client)
                relev = relevance_score(question, qr.answer, openai_client)

            result = {
                "question": question,
                "expected_sources": expected_sources,
                "retrieved_chunk_ids": retrieved_ids[:k],
                "hit_rate": hr,
                "recall": rc,
                "faithfulness": round(faith, 4),
                "relevance": round(relev, 4),
                "answer": qr.answer,
                "error": qr.error,
            }
        except Exception as exc:
            logger.error("Eval item failed: %s", exc)
            result = {
                "question": question,
                "expected_sources": expected_sources,
                "retrieved_chunk_ids": [],
                "hit_rate": 0.0,
                "recall": 0.0,
                "faithfulness": 0.0,
                "relevance": 0.0,
                "answer": "",
                "error": str(exc),
            }
        results.append(result)

    # Aggregate
    n = len(results)
    avg_hit_rate = sum(r["hit_rate"] for r in results) / n if n else 0
    avg_recall = sum(r["recall"] for r in results) / n if n else 0
    avg_faith = sum(r["faithfulness"] for r in results) / n if n else 0
    avg_relev = sum(r["relevance"] for r in results) / n if n else 0

    summary = {
        "run_id": run_id,
        "dataset": str(dataset_path),
        "n_questions": n,
        "k": k,
        "avg_hit_rate_at_k": round(avg_hit_rate, 4),
        "avg_recall_at_k": round(avg_recall, 4),
        "avg_faithfulness": round(avg_faith, 4),
        "avg_relevance": round(avg_relev, 4),
        "thresholds": {
            "hit_rate": HIT_RATE_THRESHOLD,
            "recall": RECALL_THRESHOLD,
        },
        "passed": avg_hit_rate >= HIT_RATE_THRESHOLD and avg_recall >= RECALL_THRESHOLD,
        "results": results,
    }

    # Write JSON report
    json_path = output_dir / f"{run_id}.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Write Markdown report
    md_lines = [
        f"# Eval Report — {run_id}",
        f"\n**Dataset**: `{dataset_path}`  ",
        f"**Questions**: {n}  **k**: {k}\n",
        "## Aggregate Metrics\n",
        f"| Metric | Score | Threshold | Status |",
        f"|--------|-------|-----------|--------|",
        f"| Hit-Rate@{k} | {avg_hit_rate:.3f} | {HIT_RATE_THRESHOLD} | {'✅' if avg_hit_rate >= HIT_RATE_THRESHOLD else '❌'} |",
        f"| Recall@{k}   | {avg_recall:.3f} | {RECALL_THRESHOLD} | {'✅' if avg_recall >= RECALL_THRESHOLD else '❌'} |",
        f"| Faithfulness  | {avg_faith:.3f} | — | — |",
        f"| Relevance     | {avg_relev:.3f} | — | — |",
        f"\n**Overall**: {'✅ PASSED' if summary['passed'] else '❌ FAILED'}\n",
        "## Per-Question Results\n",
    ]
    for j, r in enumerate(results, 1):
        status = "✅" if r["hit_rate"] > 0 else "❌"
        md_lines.append(
            f"### Q{j} {status} `{r['question'][:80]}`\n"
            f"- Hit-Rate: {r['hit_rate']} | Recall: {r['recall']} | "
            f"Faithfulness: {r['faithfulness']} | Relevance: {r['relevance']}\n"
            f"- **Answer**: {r['answer'][:200]}{'…' if len(r['answer']) > 200 else ''}\n"
        )

    md_path = output_dir / f"{run_id}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"\nEval run: {run_id}")
    print(f"  Hit-Rate@{k}: {avg_hit_rate:.3f}  (threshold: {HIT_RATE_THRESHOLD})")
    print(f"  Recall@{k}:   {avg_recall:.3f}  (threshold: {RECALL_THRESHOLD})")
    print(f"  Faithfulness: {avg_faith:.3f}")
    print(f"  Relevance:    {avg_relev:.3f}")
    print(f"  Status:       {'PASSED ✅' if summary['passed'] else 'FAILED ❌'}")
    print(f"\n  JSON  → {json_path}")
    print(f"  MD    → {md_path}")

    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m eval.run <dataset.json>")
        sys.exit(1)
    sys.exit(run_eval(Path(sys.argv[1])))
