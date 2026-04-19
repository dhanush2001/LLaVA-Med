# LLaVA-Med + Sinkhorn-Constrained Residual Mixing (mHC) on PathVQA

This repository extends [Microsoft LLaVA-Med v1.5](https://github.com/microsoft/LLaVA-Med) by integrating a
**Multimodal Hierarchical Classifier (mHC)** for biomedical Visual Question Answering on the **PathVQA** dataset.

The mHC adds hierarchical classification heads (mhcmlp + mhcattn) on top of the LLaVA-Med vision-language backbone,
enabling richer cross-modal reasoning for pathology images.

---

## 1. Environment Setup

```bash
conda create -n llava-med python=3.10 -y
conda activate llava-med
cd LLaVA-Med
pip install -e .
pip install peft bitsandbytes shortuuid deepspeed
```

---

## 2. Dataset: PathVQA

Place data under the following structure:

```
data/
└── pathvqa/
    ├── train.json
    ├── val.json
    ├── test.json
    └── images/
        ├── train/
        ├── val/
        └── test/
```

The JSON files are lists of conversation-style entries with fields:
`id`, `image`, `answer_type`, `conversations` (human + gpt turns)

### Convert test.json to JSONL for eval

The eval script expects one JSON object per line (JSONL), not a JSON array.
Run this once to generate `test_questions.jsonl`:

```python
python -c "
import json
data = json.load(open('data/pathvqa/test.json'))
with open('data/pathvqa/test_questions.jsonl', 'w') as f:
    for item in data:
        q = item['conversations'][0]['value'].replace('<image>', '').strip()
        entry = {
            'question_id': item['id'],
            'image':       item['image'],
            'text':        q,
            'answer_type': item.get('answer_type', 'open'),
            'gt_answer':   item['conversations'][1]['value'],
        }
        f.write(json.dumps(entry) + '\n')
print('Done:', len(data), 'entries')
"
```

---

## 3. Training

Both runs use DeepSpeed ZeRO-2, LoRA (r=128, alpha=256), frozen backbone, and tune the
`mm_projector` adapter.

### 3a. Baseline Fine-tune

Fine-tunes LLaVA-Med on PathVQA with standard LoRA + MLP adapter tuning. No mHC modules.

```bash
deepspeed llava/train/train.py \
  --deepspeed scripts/zero2.json \
  --model_name_or_path microsoft/llava-med-v1.5-mistral-7b \
  --data_path data/pathvqa/train.json \
  --image_folder data/pathvqa \
  --vision_tower openai/clip-vit-large-patch14-336 \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --bf16 True \
  --output_dir ./checkpoints/llava-med-mistral-baseline-pathvqa \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --save_strategy epoch \
  --freeze_backbone True \
  --tune_mm_mlp_adapter True \
  --lora_enable True \
  --lora_r 128 \
  --lora_alpha 256
```

### 3b. mHC Fine-tune

Same as baseline but enables the mHC residual mixing module. Uses a higher learning rate (1e-4)
and warmup (0.05) to allow the new `mHCResidual` layers (`log_W`, `stream_logits`) to converge
alongside the LoRA weights.

```bash
deepspeed llava/train/train.py \
  --deepspeed scripts/zero2.json \
  --model_name_or_path microsoft/llava-med-v1.5-mistral-7b \
  --data_path data/pathvqa/train.json \
  --image_folder data/pathvqa \
  --vision_tower openai/clip-vit-large-patch14-336 \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --bf16 True \
  --output_dir ./checkpoints/llava-med-mistral-mhc-pathvqa \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.05 \
  --lr_scheduler_type cosine \
  --max_grad_norm 1.0 \
  --save_strategy epoch \
  --freeze_backbone True \
  --tune_mm_mlp_adapter True \
  --lora_enable True \
  --lora_r 128 \
  --lora_alpha 256 \
  --mhc_enable True
```

Monitor training loss anytime:

```bash
grep '"loss"' ./checkpoints/llava-med-mistral-mhc-pathvqa/trainer_state.json | tail -10
```

---

## 4. Evaluation (Inference)

> **Important:** Checkpoint folder names must contain the word `"mistral"` so that `builder.py`
> correctly routes them through `LlavaMistralForCausalLM` instead of `AutoModelForCausalLM`.

### Baseline

```bash
PYTHONPATH=. python llava/eval/model_vqa.py \
  --model-path ./checkpoints/llava-med-mistral-baseline-pathvqa \
  --model-base mistralai/Mistral-7B-v0.1 \
  --question-file data/pathvqa/test_questions.jsonl \
  --image-folder data/pathvqa \
  --answers-file results/baseline_answers.jsonl \
  --conv-mode mistral_instruct
```

### mHC Model

```bash
PYTHONPATH=. python llava/eval/model_vqa.py \
  --model-path ./checkpoints/llava-med-mistral-mhc-pathvqa \
  --model-base mistralai/Mistral-7B-v0.1 \
  --question-file data/pathvqa/test_questions.jsonl \
  --image-folder data/pathvqa \
  --answers-file results/mhc_answers.jsonl \
  --conv-mode mistral_instruct
```

Check progress:

```bash
wc -l results/baseline_answers.jsonl
wc -l results/mhc_answers.jsonl
```

---

## 5. Scoring

```bash
python llava/eval/eval_vqa.py \
  --pred-file results/baseline_answers.jsonl \
  --anno-file data/pathvqa/test_questions.jsonl

python llava/eval/eval_vqa.py \
  --pred-file results/mhc_answers.jsonl \
  --anno-file data/pathvqa/test_questions.jsonl
```

---

## 6. Library Changes Made

### 6a. `llava/model/builder.py` — Tokenizer Fix

**Problem:** `AutoTokenizer.from_pretrained(model_path)` raises:
```
KeyError: 'LlavaMistralConfig'
```
because HuggingFace's `AutoTokenizer` does not know about the custom `LlavaMistralConfig`
registered in this repo.

**Fix:** When loading a Mistral-based LLaVA checkpoint, load the tokenizer from the
base model (`model_base`, e.g. `mistralai/Mistral-7B-v0.1`) instead of from the checkpoint
path. The model weights still load from `model_path` via `LlavaMistralForCausalLM`.

```python
# Before (broken):
tokenizer = AutoTokenizer.from_pretrained(model_path)

# After (fixed):
tok_source = model_base if model_base else model_path
tokenizer = AutoTokenizer.from_pretrained(tok_source, use_fast=False)
```

### 6b. `llava/train/train.py` — mHC Unfreeze Fix

**Problem:** After LoRA initialization with `freeze_backbone=True`, the new `mHCResidual`
parameters (`log_W`, `stream_logits`) were also frozen, causing `loss=0.0` throughout
training since no gradients flowed through the residual mixing layers.

**Fix:** After LoRA wrapping, explicitly unfreeze any parameter belonging to an `mHCResidual`
module. The parameters to unfreeze are `log_W` and `stream_logits` — identifiable by the
`mhc_residual` name prefix used in `llava_mistral.py`:

```python
for name, param in model.named_parameters():
    if 'mhc_residual' in name:
        param.requires_grad = True
```

> **Note:** If your integration names the modules differently (e.g. `mhcmlp`, `mhcattn`),
> update the string match accordingly. The key point is that all `mHCResidual` parameters
> must have `requires_grad = True` after LoRA wrapping.

### 6c. `data/pathvqa/test_questions.jsonl` — Format Conversion

**Problem:** The PathVQA `test.json` is a JSON array. The eval script (`model_vqa.py`) reads
the file line-by-line with `json.loads()`, expecting JSONL format (one object per line).

**Fix:** Convert once using the script in Section 2. Fields mapped:

| Source | Destination |
|--------|-------------|
| `id` | `question_id` |
| `image` | `image` (already includes `images/test/` prefix) |
| `conversations[0].value` | `text` (human question, stripped of `<image>` token) |
| `conversations[1].value` | `gt_answer` (stored for scoring) |

---

## 7. Repository Structure

```
LLaVA-Med/
├── llava/
│   ├── model/
│   │   ├── builder.py                  # MODIFIED: tokenizer loading fix
│   │   └── language_model/
│   │       └── llava_mistral.py        # mHC integration (mHCResidual)
│   ├── train/
│   │   └── train.py                    # MODIFIED: mHC unfreeze fix
│   └── eval/
│       └── model_vqa.py                # Inference script (unchanged)
├── mhc.py                              # mHCResidual module (Sinkhorn + doubly stochastic mixing)
├── data/
│   └── pathvqa/
│       ├── train.json                  # Training data
│       ├── test.json                   # Raw test data
│       └── test_questions.jsonl        # GENERATED: use for eval
├── checkpoints/                        # Model weights (git-ignored)
├── results/                            # Eval outputs (git-ignored)
└── scripts/
    └── zero2.json                      # DeepSpeed ZeRO-2 config
```

---

## 8. Citation

```bibtex
@article{li2023llava-med,
  title={LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day},
  author={Li, Chunyuan and others},
  booktitle={NeurIPS},
  year={2023}
}

@article{xie2025mhc,
  title={mHC: Manifold-Constrained Hyper-Connections},
  author={Xie, Ziyang and others},
  journal={arXiv:2512.24880},
  year={2025}
}
```
