"""
eval/metrics.py — Retrieval and generation quality metrics.

Retrieval metrics (ground-truth-based):
  hit_rate_at_k   — was any expected source in the top-k results?
  recall_at_k     — fraction of expected sources found in top-k

Generation proxy metrics (LLM-judge):
  faithfulness_score  — is the answer grounded in the provided context?
  relevance_score     — does the answer address the question?
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


# ── Retrieval metrics ──────────────────────────────────────────────────────────

def hit_rate_at_k(
    retrieved_ids: List[str],
    expected_ids: List[str],
    k: int,
) -> float:
    """
    1.0 if at least one expected source appears in the top-k retrieved IDs,
    0.0 otherwise.

    IDs can be chunk_ids or doc_ids — just be consistent.
    """
    if not expected_ids:
        return 1.0  # no ground truth → treat as pass
    top_k = set(retrieved_ids[:k])
    return 1.0 if top_k & set(expected_ids) else 0.0


def recall_at_k(
    retrieved_ids: List[str],
    expected_ids: List[str],
    k: int,
) -> float:
    """
    Fraction of expected sources found in the top-k retrieved IDs.
    Returns 1.0 when expected_ids is empty.
    """
    if not expected_ids:
        return 1.0
    top_k = set(retrieved_ids[:k])
    found = top_k & set(expected_ids)
    return len(found) / len(expected_ids)


# ── Generation proxy metrics (LLM-judge) ──────────────────────────────────────

_FAITHFULNESS_PROMPT = """\
You are an evaluation judge. Given a CONTEXT and an ANSWER, rate how well the
answer is grounded in the context on a scale from 0.0 (completely unsupported)
to 1.0 (fully supported by context).

Respond with ONLY a single float number and nothing else.

CONTEXT:
{context}

ANSWER:
{answer}
"""

_RELEVANCE_PROMPT = """\
You are an evaluation judge. Given a QUESTION and an ANSWER, rate how well
the answer addresses the question on a scale from 0.0 (completely irrelevant)
to 1.0 (perfectly answers the question).

Respond with ONLY a single float number and nothing else.

QUESTION:
{question}

ANSWER:
{answer}
"""


def _llm_score(prompt: str, client, model: str = "gpt-4o-mini") -> float:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        text = resp.choices[0].message.content.strip()
        return max(0.0, min(1.0, float(text)))
    except Exception as exc:
        logger.warning("LLM judge failed: %s", exc)
        return 0.5  # neutral fallback


def faithfulness_score(
    answer: str,
    context: str,
    client,
    model: str = "gpt-4o-mini",
) -> float:
    """LLM-judge: is the answer supported by the retrieved context? (0–1)"""
    prompt = _FAITHFULNESS_PROMPT.format(context=context[:3000], answer=answer[:1500])
    return _llm_score(prompt, client, model)


def relevance_score(
    question: str,
    answer: str,
    client,
    model: str = "gpt-4o-mini",
) -> float:
    """LLM-judge: does the answer address the question? (0–1)"""
    prompt = _RELEVANCE_PROMPT.format(question=question, answer=answer[:1500])
    return _llm_score(prompt, client, model)
