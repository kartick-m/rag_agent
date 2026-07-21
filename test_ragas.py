# from langchain_community.chat_models

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from app import graph
GOLDEN_SET = [
    {
        "question": "What is Ownership Transition Risk?",
        "ground_truth": (
            "Ownership Transition Risk refers to the risk businesses face "
            "when leadership or ownership changes hands, particularly "
            "during the wave of business owner retirements and succession "
            "gaps described in the McKinsey report."
        ),
    },
]

THRESHOLDS = {
    "faithfulness": 0.75,
    "answer_relevancy": 0.80,
    "context_precision": 0.70,
    "context_recall": 0.80,
}


def collect_records(golden_set):
    records = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in golden_set:
        result = graph.invoke({"user_prompt": item["question"]})
 
        if not result.get("input_safe", True):
            print(f"[skip] guard blocked golden question: {item['question']!r}")
            continue
 
        records["question"].append(item["question"])
        records["answer"].append(result["llm_response"])
        records["contexts"].append(result["retrieved_contexts"])
        records["ground_truth"].append(item["ground_truth"])
    return records
 
 
def main():
    records = collect_records(GOLDEN_SET)
    dataset = Dataset.from_dict(records)
 
    report = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
 
    df = report.to_pandas()
    print(df.to_string(index=False))
 
    print("\n--- Threshold check ---")
    failures = []
    for metric, threshold in THRESHOLDS.items():
        avg = df[metric].mean()
        status = "PASS" if avg >= threshold else "FAIL"
        print(f"{metric:20s} avg={avg:.2f}  threshold={threshold}  [{status}]")
        if avg < threshold:
            failures.append(metric)
 
    if failures:
        raise SystemExit(f"Regression detected in: {', '.join(failures)}")
 
 
if __name__ == "__main__":
    main()
