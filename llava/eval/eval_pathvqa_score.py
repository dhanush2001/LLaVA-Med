"""
Score PathVQA predictions produced by llava/eval/model_vqa.py.

Reports Overall / Open-Ended / Close-Ended accuracy, matching the metric used
in the project README:
  - Close-ended (yes/no): yes/no polarity match after normalization (1.0 / 0.0)
  - Open-ended:           configurable via --open-metric (default: recall)
  - Overall:              mean of every sample's per-question score

Questions present in the annotations but missing from the predictions are
scored 0 (not dropped), so partial-coverage runs are comparable.

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
    """Lowercase, replace punctuation with spaces, drop articles, collapse whitespace.
    Punctuation becomes a space (not deleted) so hyphenated/slashed terms like
    'T-cell' split into ['t', 'cell'] and match a spaced 'T cell'."""
    text = text.lower().strip()
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    tokens = [t for t in re.split(r"\s+", text) if t and t not in ARTICLES]
    return tokens


def is_closed(answer_type, gt_norm):
    """PathVQA closed questions are yes/no. Trust answer_type when it is present
    (so an open question whose answer happens to be 'yes'/'no' stays open); only
    fall back to detecting a yes/no gold when answer_type is absent."""
    if answer_type:
        return str(answer_type).lower() in {"closed", "close", "yes/no"}
    return gt_norm in (["yes"], ["no"])


def score_closed(gt_norm, pred_norm):
    """Score a closed question by polarity, not substring membership. Closed golds
    are almost always yes/no and the model typically leads with the answer
    ('Yes, it is'), so match the first token; otherwise credit an unambiguous
    yes/no elsewhere in the answer. A prediction containing both (or neither) is
    wrong. For a rare non-yes/no closed gold, require exact match."""
    if gt_norm in (["yes"], ["no"]):
        gold = gt_norm[0]
        if pred_norm[:1] in (["yes"], ["no"]):
            return 1.0 if pred_norm[0] == gold else 0.0
        has_yes, has_no = "yes" in pred_norm, "no" in pred_norm
        if has_yes ^ has_no:
            return 1.0 if (gold == "yes") == has_yes else 0.0
        return 0.0
    return 1.0 if pred_norm == gt_norm else 0.0


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
        gt_norm = normalize(a.get("gt_answer", ""))
        if qid in preds:
            pred_norm = normalize(preds[qid])
        else:
            # No prediction -> score wrong (empty pred) rather than dropping it,
            # so a partial-coverage run doesn't get an inflated denominator.
            missing += 1
            pred_norm = []

        if is_closed(a.get("answer_type"), gt_norm):
            score = score_closed(gt_norm, pred_norm)
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
          + (f"; {missing} missing predictions scored 0" if missing else "") + ")")
    print("-" * 48)
    print(f"Overall     : {pct(all_scores):6.2f}%")
    print(f"Open-Ended  : {pct(open_scores):6.2f}%   (open-metric: {args.open_metric})")
    print(f"Close-Ended : {pct(close_scores):6.2f}%   (yes/no match)")


if __name__ == "__main__":
    main()
