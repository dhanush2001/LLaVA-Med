from datasets import load_dataset
import json, os
from PIL import Image

def save_split(ds, split, out_dir):
    if split not in ds:
        print(f"  Skipping {split} (not found)")
        return
    img_dir = f"{out_dir}/images/{split}"
    os.makedirs(img_dir, exist_ok=True)
    records = []
    for i, s in enumerate(ds[split]):
        img_path = f"{img_dir}/{i}.jpg"
        if isinstance(s["image"], Image.Image):
            s["image"].save(img_path)
        records.append({
            "id": f"{split}_{i}",
            "image": img_path,
            "answer_type": "closed" if str(s["answer"]).lower() in ["yes","no"] else "open",
            "conversations": [
                {"from": "human", "value": f"<image>\n{s['question']}"},
                {"from": "gpt",   "value": str(s["answer"])}
            ]
        })
    out_path = f"{out_dir}/{split}.json"
    with open(out_path, "w") as f:
        json.dump(records, f)
    print(f"  ✅ {split}: {len(records)} samples → {out_path}")

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
