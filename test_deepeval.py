"""
DeepEval RAG quality tests for the Agent-Guard graph.

These sit alongside your existing test_app.py, which checks guard
behavior (safe/blocked). This file checks *answer quality* on the
happy path: given a safe prompt, is the retrieved context relevant,
is the answer faithful to it, and is it actually relevant to the
question.

Run:
    pytest test_rag_deepeval.py -m integration
    # or, for DeepEval's own CLI/reporting:
    deepeval test run test_rag_deepeval.py
"""

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)

from app import graph

# Reuse one judge model config across metrics rather than repeating
# it four times — swap for whatever model your org standardizes on.
JUDGE_MODEL = "gpt-4o-mini"

faithfulness = FaithfulnessMetric(threshold=0.7, model=JUDGE_MODEL)
answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model=JUDGE_MODEL)
contextual_precision = ContextualPrecisionMetric(threshold=0.7, model=JUDGE_MODEL)
contextual_recall = ContextualRecallMetric(threshold=0.7, model=JUDGE_MODEL)

# Golden set: real questions + an SME-approved reference answer.
# expected_output is required for ContextualPrecision/Recall — without
# it those two metrics can't be computed, so don't skip authoring these.
RAG_GOLDEN_SET = [
    {
        "user_prompt": "What is Ownership Transition Risk?",
        "expected_output": (
            "Ownership Transition Risk refers to the risk businesses face "
            "when leadership or ownership changes hands, particularly "
            "during the wave of business owner retirements and succession "
            "gaps described in the McKinsey report."
        ),
    },
    # Add more (question, reference answer) pairs as your corpus grows.
    # Aim for coverage of: easy factual lookups, multi-chunk synthesis
    # questions, and edge cases where the corpus doesn't fully answer it.
]


@pytest.mark.integration
@pytest.mark.parametrize("case", RAG_GOLDEN_SET, ids=lambda c: c["user_prompt"][:40])
def test_rag_answer_quality(case):
    result = graph.invoke({"user_prompt": case["user_prompt"]})

    # Sanity check first — if the guard blocked a prompt that should
    # have gone through, fail fast with a clear message rather than
    # letting the RAG metrics fail confusingly on a "Prompt Rejected" output.
    assert result["input_safe"] is True, (
        f"Input guard blocked a prompt expected to pass: {case['user_prompt']!r}"
    )

    test_case = LLMTestCase(
        input=case["user_prompt"],
        actual_output=result["llm_response"],
        retrieval_context=result["retrieved_contexts"],
        expected_output=case["expected_output"],
    )

    assert_test(
        test_case,
        [faithfulness, answer_relevancy, contextual_precision, contextual_recall],
    )