"""
Score PathVQA predictions produced by llava/eval/model_vqa.py.

Reports Overall / Open-Ended / Close-Ended accuracy, matching the metric used
in the project README:
  - Close-ended (yes/no): exact match after normalization (1.0 / 0.0)
  - Open-ended:           configurable via --open-metric (default: recall)
  - Overall:              mean of every sample's per-question score

--open-metric choices:
  recall   token recall = |gt_tokens ∩ pred_tokens| / |gt_tokens|   (partial credit)
  contains 1.0 if every gt token appears in the prediction, else 0.0
  exact    1.0 only if the normalized prediction equals the normalized answer

Usage:
  python llava/eval/eval_pathvqa_score.py \
      --pred-file results/mhc_answers.jsonl \
      --anno-file data/pathvqa/test_questions.jsonl \
      --open-metric recall
"""
import argparse
import json
import re
import string


ARTICLES = {"a", "an", "the"}


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize(text):
    """Lowercase, drop punctuation and articles, collapse whitespace."""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in re.split(r"\s+", text) if t and t not in ARTICLES]
    return tokens


def is_closed(answer_type, gt_norm):
    """PathVQA closed questions are yes/no."""
    if answer_type and str(answer_type).lower() in {"closed", "close", "yes/no"}:
        return True
    return gt_norm in (["yes"], ["no"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-file", required=True,
                    help="model_vqa.py output (question_id, text)")
    ap.add_argument("--anno-file", required=True,
                    help="test_questions.jsonl (question_id, gt_answer, answer_type)")
    ap.add_argument("--open-metric", choices=["recall", "contains", "exact"],
                    default="recall",
                    help="scoring for open-ended questions (default: recall)")
    args = ap.parse_args()

    preds = {p["question_id"]: p.get("text", "") for p in load_jsonl(args.pred_file)}
    annos = load_jsonl(args.anno_file)

    open_scores, close_scores, all_scores = [], [], []
    missing = 0

    for a in annos:
        qid = a["question_id"]
        if qid not in preds:
            missing += 1
            continue
        gt_norm = normalize(a.get("gt_answer", ""))
        pred_norm = normalize(preds[qid])

        if is_closed(a.get("answer_type"), gt_norm):
            score = 1.0 if gt_norm == pred_norm or (gt_norm and gt_norm[0] in pred_norm) else 0.0
            close_scores.append(score)
        else:
            if not gt_norm:
                continue
            if args.open_metric == "exact":
                score = 1.0 if pred_norm == gt_norm else 0.0
            elif args.open_metric == "contains":
                score = 1.0 if all(t in set(pred_norm) for t in gt_norm) else 0.0
            else:  # recall
                hits = sum(1 for t in gt_norm if t in set(pred_norm))
                score = hits / len(gt_norm)
            open_scores.append(score)
        all_scores.append(score)

    def pct(xs):
        return 100.0 * sum(xs) / len(xs) if xs else 0.0

    print(f"Predictions file : {args.pred_file}")
    print(f"Annotations file : {args.anno_file}")
    print(f"Scored {len(all_scores)} questions "
          f"({len(close_scores)} closed, {len(open_scores)} open"
          + (f", {missing} missing predictions" if missing else "") + ")")
    print("-" * 48)
    print(f"Overall     : {pct(all_scores):6.2f}%")
    print(f"Open-Ended  : {pct(open_scores):6.2f}%   (open-metric: {args.open_metric})")
    print(f"Close-Ended : {pct(close_scores):6.2f}%   (exact match)")


if __name__ == "__main__":
    main()
