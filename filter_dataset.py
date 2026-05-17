# run once: filter_dataset.py
import json

def count_words(record):
    msgs = record.get("messages", [])
    total = sum(len(m["content"].split()) for m in msgs)
    return total

for filename in ["train.jsonl", "valid.jsonl"]:
    path = f"./skillvault_mlx/{filename}"
    records = []
    skipped = 0

    with open(path) as f:
        for line in f:
            r = json.loads(line.strip())
            # ~1024 tokens ≈ 700 words (rough estimate)
            if count_words(r) <= 700:
                records.append(r)
            else:
                skipped += 1

    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"{filename}: kept {len(records)}, removed {skipped} long sequences")