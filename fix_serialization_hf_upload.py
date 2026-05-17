"""
fix_and_reupload.py
====================
Fixes the ArrowNotImplementedError: '_format_kwargs' empty struct issue
by rebuilding a clean dataset from the raw JSONL files and re-uploading.

Usage:
    python fix_and_reupload.py --repo ekaansh/skillvault --token hf_yourtoken
"""

import json
import argparse
from pathlib import Path
from datasets import Dataset, DatasetDict


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ⚠️  Skipping malformed line {i+1}: {e}")
    return records


def clean_record(record: dict) -> dict:
    """
    Ensure every value is a plain Python type.
    Converts None → "" for string fields, removes any nested dicts
    that could cause struct serialization issues.
    """
    clean = {}
    for key, value in record.items():
        if value is None:
            clean[key] = ""
        elif isinstance(value, list):
            # Flatten lists of dicts to JSON strings (safe for Parquet)
            clean[key] = json.dumps(value) if value and isinstance(value[0], dict) else value
        elif isinstance(value, dict):
            # Convert nested dicts to JSON string — Parquet can't handle empty structs
            clean[key] = json.dumps(value)
        else:
            clean[key] = value
    return clean


def build_clean_dataset(jsonl_path: str) -> DatasetDict:
    """
    Load JSONL → clean records → fresh Dataset → train/val split.
    No save_to_disk() metadata, no _format_kwargs, pure Arrow-safe data.
    """
    print(f"\n📥 Loading: {jsonl_path}")
    records = load_jsonl(jsonl_path)
    print(f"   Loaded {len(records)} records")

    # Clean every record
    cleaned = [clean_record(r) for r in records]

    # Verify all records have the same keys
    all_keys = set()
    for r in cleaned:
        all_keys.update(r.keys())

    # Fill missing keys with empty string
    for r in cleaned:
        for key in all_keys:
            if key not in r:
                r[key] = ""

    print(f"   Columns: {sorted(all_keys)}")

    # Build fresh Dataset directly from list of dicts
    # This bypasses save_to_disk() metadata entirely
    ds = Dataset.from_list(cleaned)

    # Train / validation split (90 / 10)
    split = ds.train_test_split(test_size=0.1, seed=42)

    print(f"\n✅ Clean dataset built:")
    print(f"   Train : {len(split['train'])} rows")
    print(f"   Val   : {len(split['test'])} rows")
    print(f"   Schema: {split['train'].features}")

    return split


def push_to_hub(dataset: DatasetDict, repo_id: str, token: str):
    """Push the clean dataset to HuggingFace Hub."""
    from huggingface_hub import login
    login(token=token)

    print(f"\n🚀 Pushing to: https://huggingface.co/datasets/{repo_id}")

    dataset.push_to_hub(
        repo_id,
        private=False,
        commit_message="Fix: rebuild clean dataset from JSONL, resolves _format_kwargs Parquet error",
    )

    print(f"✅ Done → https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Fix and re-upload SkillVault dataset")
    parser.add_argument("--repo",  "-r", required=True, help="HuggingFace repo ID (e.g. ekaansh/skillvault)")
    parser.add_argument("--token", "-t", required=True, help="HuggingFace write token")
    parser.add_argument(
        "--jsonl", "-j",
        default="./skillvault_data/training_pairs.jsonl",
        help="Path to training_pairs.jsonl (default: ./skillvault_data/training_pairs.jsonl)"
    )
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"❌ File not found: {jsonl_path}")
        print("   Make sure you're running from the folder containing skillvault_data/")
        return

    # Build clean dataset
    dataset = build_clean_dataset(str(jsonl_path))

    # Push
    push_to_hub(dataset, args.repo, args.token)


if __name__ == "__main__":
    main()