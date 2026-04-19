# LLaVA-Med + Multimodal Hierarchical Classifier (mHC) on PathVQA

This repository extends [Microsoft LLaVA-Med v1.5](https://github.com/microsoft/LLaVA-Med) by integrating a
**Manifold-Constrained Hyper-Connections (mHC)** for biomedical Visual Question Answering on the **PathVQA** dataset.

---

## 1. Environment Setup

    conda create -n llava-med python=3.10 -y
    conda activate llava-med
    cd LLaVA-Med
    pip install -e .
    pip install peft bitsandbytes shortuuid deepspeed

---

## 2. Dataset: PathVQA

Place data under the following structure:

    data/
    └── pathvqa/
        ├── train.json
        ├── val.json
        ├── test.json
        └── images/
            ├── train/
            ├── val/
            └── test/

The JSON files are lists of conversation-style entries with fields:
- id, image, answer_type, conversations (human + gpt turns)

### Convert test.json to JSONL for eval

The eval script expects one JSON object per line (JSONL), not a JSON array.
Run this once to generate test_questions.jsonl:

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

---

## 3. Training

Both runs use DeepSpeed ZeRO-2, LoRA (r=128, alpha=256), frozen backbone, and tune the mm_projector adapter.

### 3a. Baseline Fine-tune

Fine-tunes LLaVA-Med on PathVQA with standard LoRA + MLP adapter tuning. No mHC modules.

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

### 3b. mHC Fine-tune

Same as baseline but enables the mHC modules. Uses a higher learning rate (1e-4) and warmup (0.05)
to allow the new mhcmlp and mhcattn layers to converge alongside the LoRA weights.

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

Monitor training loss anytime:

    grep '"loss"' ./checkpoints/llava-med-mistral-mhc-pathvqa/trainer_state.json | tail -10

---

## 4. Evaluation (Inference)

IMPORTANT: Checkpoint folder names must contain the word "mistral" so that builder.py
correctly routes them through LlavaMistralForCausalLM instead of AutoModelForCausalLM.

### Baseline

    PYTHONPATH=. python llava/eval/model_vqa.py \
      --model-path ./checkpoints/llava-med-mistral-baseline-pathvqa \
      --model-base mistralai/Mistral-7B-v0.1 \
      --question-file data/pathvqa/test_questions.jsonl \
      --image-folder data/pathvqa \
      --answers-file results/baseline_answers.jsonl \
      --conv-mode vicuna_v1

### mHC Model

    PYTHONPATH=. python llava/eval/model_vqa.py \
      --model-path ./checkpoints/llava-med-mistral-mhc-pathvqa \
      --model-base mistralai/Mistral-7B-v0.1 \
      --question-file data/pathvqa/test_questions.jsonl \
      --image-folder data/pathvqa \
      --answers-file results/mhc_answers.jsonl \
      --conv-mode vicuna_v1

Check progress:

    wc -l results/baseline_answers.jsonl
    wc -l results/mhc_answers.jsonl

---

## 5. Scoring

    python llava/eval/eval_vqa.py \
      --pred-file results/baseline_answers.jsonl \
      --anno-file data/pathvqa/test_questions.jsonl

    python llava/eval/eval_vqa.py \
      --pred-file results/mhc_answers.jsonl \
      --anno-file data/pathvqa/test_questions.jsonl

---

## 6. Library Changes Made

### 6a. llava/model/builder.py — Tokenizer Fix

PROBLEM: AutoTokenizer.from_pretrained(model_path) raises:
    KeyError: 'LlavaMistralConfig'
because HuggingFace's AutoTokenizer does not know about the custom LlavaMistralConfig
registered in this repo.

FIX: When loading a mistral-based LLaVA checkpoint, load the tokenizer from the
base model (model_base argument, e.g. mistralai/Mistral-7B-v0.1) instead of from the
checkpoint path. The model weights still load from model_path via LlavaMistralForCausalLM.

    # Before (broken):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # After (fixed):
    tok_source = model_base if model_base else model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_source, use_fast=False)

### 6b. llava/train/train.py — mHC Unfreeze Fix

PROBLEM: After LoRA initialization with freeze_backbone=True, the new mHC modules
(mhcmlp, mhcattn) were also frozen, causing loss=0.0 throughout training since no
gradients flowed through the classification heads.

FIX: After LoRA wrapping, explicitly unfreeze any parameter whose name contains
'mhcmlp' or 'mhcattn':

    for name, param in model.named_parameters():
        if 'mhcmlp' in name or 'mhcattn' in name:
            param.requires_grad = True

This ensures the mHC layers train while the backbone LLM weights remain frozen.

### 6c. data/pathvqa/test_questions.jsonl — Format Conversion

PROBLEM: The PathVQA test.json is a JSON array. The eval script (model_vqa.py) reads
the file line-by-line with json.loads(), expecting JSONL format (one object per line).

FIX: Convert once using the script in Section 2. Fields mapped:
    id              -> question_id
    image           -> image  (already includes images/test/ prefix)
    conversations[0].value -> text (human question, stripped of <image> token)
    conversations[1].value -> gt_answer (stored for scoring)

---

## 7. Repository Structure

    LLaVA-Med/
    ├── llava/
    │   ├── model/
    │   │   ├── builder.py                  # MODIFIED: tokenizer loading fix
    │   │   └── language_model/
    │   │       └── llava_mistral.py        # mHC integration
    │   ├── train/
    │   │   └── train.py                    # MODIFIED: mHC unfreeze fix
    │   └── eval/
    │       └── model_vqa.py                # Inference script (unchanged)
    ├── data/
    │   └── pathvqa/
    │       ├── train.json                  # Training data
    │       ├── test.json                   # Raw test data
    │       └── test_questions.jsonl        # GENERATED: use for eval
    ├── checkpoints/                        # Model weights (git-ignored)
    ├── results/                            # Eval outputs (git-ignored)
    └── scripts/
        └── zero2.json                      # DeepSpeed ZeRO-2 config
