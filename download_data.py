import json
import os

import numpy as np
from datasets import load_dataset
from PIL import Image


def _save_image(sample_image, img_path):
    """Save image payloads from multiple HF dataset formats."""
    if isinstance(sample_image, Image.Image):
        sample_image.convert("RGB").save(img_path)
        return

    if isinstance(sample_image, dict):
        # Common datasets.Image payload format.
        if "path" in sample_image and sample_image["path"]:
            Image.open(sample_image["path"]).convert("RGB").save(img_path)
            return
        if "bytes" in sample_image and sample_image["bytes"]:
            from io import BytesIO
            Image.open(BytesIO(sample_image["bytes"])).convert("RGB").save(img_path)
            return

    # Fallback for array-like image objects.
    Image.fromarray(np.array(sample_image)).convert("RGB").save(img_path)

def save_split(ds, split, out_dir):
    if split not in ds:
        print(f"  Skipping {split} (not found)")
        return
    img_dir = f"{out_dir}/images/{split}"
    os.makedirs(img_dir, exist_ok=True)
    records = []
    skipped = 0
    for i, s in enumerate(ds[split]):
        img_path = f"{img_dir}/{i}.jpg"
        try:
            _save_image(s["image"], img_path)
        except Exception as exc:
            skipped += 1
            print(f"  Skipping {split}_{i}: image save failed ({exc})")
            continue

        answer = str(s.get("answer", "")).strip()
        if not answer:
            skipped += 1
            print(f"  Skipping {split}_{i}: empty answer")
            continue

        question = str(s.get("question", "")).strip()
        if not question:
            skipped += 1
            print(f"  Skipping {split}_{i}: empty question")
            continue

        records.append({
            "id": f"{split}_{i}",
            "image": img_path,
            "answer_type": "closed" if answer.lower() in ["yes", "no"] else "open",
            "conversations": [
                {"from": "human", "value": f"<image>\n{question}"},
                {"from": "gpt", "value": answer}
            ]
        })
    out_path = f"{out_dir}/{split}.json"
    with open(out_path, "w") as f:
        json.dump(records, f, ensure_ascii=True)
    print(f"  ✅ {split}: {len(records)} samples ({skipped} skipped) → {out_path}")

print("📥 Downloading PathVQA (~785MB)...")
pvqa = load_dataset("flaviagiammarino/path-vqa")
os.makedirs("data/pathvqa", exist_ok=True)
for split in ["train", "validation", "test"]:
    save_split(pvqa, split, "data/pathvqa")

print("\n📥 Downloading VQA-RAD (~50MB)...")
vrad = load_dataset("flaviagiammarino/vqa-rad")
os.makedirs("data/vqarad", exist_ok=True)
for split in ["train", "test"]:
    save_split(vrad, split, "data/vqarad")

print("\n✅ All datasets ready.")
