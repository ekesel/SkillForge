# convert_dataset.py
# Run this once to convert training_pairs.jsonl → MLX format

import json
from pathlib import Path

input_path  = "./skillvault_data/training_pairs.jsonl"
output_path = "./skillvault_mlx/"

Path(output_path).mkdir(exist_ok=True)

SYSTEM_PROMPT = """You are SkillForge, an expert AI that generates professional \
agent skill files (SKILL.md). You deeply understand how AI coding agents like \
Claude Code, Cursor, Gemini CLI, and Codex use skill files to perform specialized \
tasks."""

def to_mlx_format(record):
    """Convert training pair to MLX chat format."""
    instruction = record["instruction"]
    user_input  = record.get("input", "")
    output      = record["output"]

    user_msg = f"{instruction}\n\n{user_input}" if user_input.strip() else instruction

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": output},
        ]
    }

# Load and convert
records = []
with open(input_path) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f"Loaded {len(records)} records")

# Shuffle and split 90/10
import random
random.seed(42)
random.shuffle(records)

split      = int(len(records) * 0.9)
train_data = records[:split]
val_data   = records[split:]

# Write train.jsonl and valid.jsonl
for filename, data in [("train.jsonl", train_data), ("valid.jsonl", val_data)]:
    with open(f"{output_path}/{filename}", "w") as f:
        for r in data:
            f.write(json.dumps(to_mlx_format(r)) + "\n")
    print(f"✅ {filename}: {len(data)} records → {output_path}")