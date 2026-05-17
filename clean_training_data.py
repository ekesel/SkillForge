"""
clean_training_data.py
======================
Cleans the SkillVault training data by removing noisy records
that cause hallucination in the fine-tuned model.

Noise sources identified from output analysis:
  - Composio SaaS API skills (mixed wrong patterns into general skills)
  - Fake URLs and library names (redfish, starapp, fastapp, excellentos)
  - Malformed YAML with empty structs
  - Records too short to be useful
  - Records with garbled/repeated content

Also fixes max_tokens in test_skillforge.py from 600 → 800.

Usage:
    python3 clean_training_data.py
"""

import json
import re
from pathlib import Path

# ─────────────────────────────────────────────
# NOISE SIGNALS
# Any record containing these strings gets removed.
# Identified from bad outputs in v1 evaluation.
# ─────────────────────────────────────────────
NOISE_KEYWORDS = [
    # Composio SaaS patterns that polluted general skills
    "composio",
    "rube mcp",
    "pipewire.github",
    "realpython.github",

    # Fake libraries found in hallucinated outputs
    "redfish",          # fake Redis library
    "starapp",          # fake FastAPI command
    "fastapp import",   # fake FastAPI library
    "excellentos",      # fake domain
    "fastapi.excellen", # fake URL

    # Broken broker configs from Celery noise
    "amq://:password",
    "default Exchange Binding Key",

    # Garbled content patterns
    "pipewire.github.io",
    "install.py\"",
    "malware-analysis",
]

# Regex patterns for structural noise
NOISE_PATTERNS = [
    # Lines with 5+ repeated identical segments
    r'(.{10,})\1{4,}',

    # Fake pip install URLs
    r'pip install -I https://',

    # Empty or near-empty outputs
    r'^.{0,30}$',
]

# ─────────────────────────────────────────────
# QUALITY FILTERS
# ─────────────────────────────────────────────
MIN_WORD_COUNT   = 15    # ← change to 15
MIN_OUTPUT_CHARS = 50   # ← change to 50
MAX_OUTPUT_CHARS = 3000  # ← keep same


def has_noise_keyword(text: str) -> bool:
    """Check if text contains any noise keyword."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in NOISE_KEYWORDS)


def has_noise_pattern(text: str) -> bool:
    """Check if text matches any noise regex pattern."""
    for pattern in NOISE_PATTERNS:
        if re.search(pattern, text, re.MULTILINE):
            return True
    return False


def has_repetition_loop(text: str) -> bool:
    """
    Detect repetition loops — same phrase repeated 4+ times.
    These cause the model to learn looping behaviour.
    """
    # Split into lines, check for repeated blocks
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) < 8:
        return False

    # Check if any 3-line block repeats 3+ times
    for i in range(len(lines) - 3):
        block = tuple(lines[i:i+3])
        count = sum(
            1 for j in range(len(lines) - 3)
            if tuple(lines[j:j+3]) == block
        )
        if count >= 3:
            return True

    # Check if any single line repeats 4+ times
    from collections import Counter
    line_counts = Counter(lines)
    if any(c >= 4 for c in line_counts.values()):
        return True

    return False


def is_quality_record(record: dict) -> tuple[bool, str]:
    """
    Returns (True, "") if record is clean.
    Returns (False, reason) if record should be removed.
    """
    # Handle both flat format {instruction, input, output}
    # and MLX chat format {messages: [{role, content}...]}
    messages = record.get("messages", [])
    if messages:
        # MLX chat format — output is the last assistant message
        output = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "assistant"),
            ""
        )
        instruction = next(
            (m["content"] for m in messages if m.get("role") == "user"),
            ""
        )
    else:
        # Flat format
        output = record.get("output", "")
        instruction = record.get("instruction", "")
    full_text = json.dumps(record)

    # ── Length checks ──
    if len(output.strip()) < MIN_OUTPUT_CHARS:
        return False, f"output too short ({len(output)} chars)"

    if len(output.split()) < MIN_WORD_COUNT:
        return False, f"output too few words ({len(output.split())} words)"

    if len(output) > MAX_OUTPUT_CHARS:
        return False, f"output too long ({len(output)} chars)"

    # ── Noise keyword check ──
    if has_noise_keyword(full_text):
        matched = [kw for kw in NOISE_KEYWORDS if kw in full_text.lower()]
        return False, f"noise keywords: {matched[:3]}"

    # ── Noise pattern check ──
    if has_noise_pattern(output):
        return False, "noise pattern detected"

    # ── Repetition loop check ──
    if has_repetition_loop(output):
        return False, "repetition loop detected"

    # ── Must have instruction ──
    if len(instruction.strip()) < 10:
        return False, "instruction too short"

    return True, ""


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def clean_file(input_path: str) -> dict:
    """Clean a single JSONL file. Returns stats dict."""
    path = Path(input_path)
    if not path.exists():
        print(f"  ⚠️  File not found: {input_path}")
        return {}

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    print(f"\n📂 {path.name}: {len(records)} records loaded")

    # Filter records
    kept, removed = [], []
    removal_reasons = {}

    for record in records:
        clean, reason = is_quality_record(record)
        if clean:
            kept.append(record)
        else:
            removed.append(record)
            removal_reasons[reason] = removal_reasons.get(reason, 0) + 1

    # Write cleaned file (overwrite)
    with open(path, "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Print removal breakdown
    print(f"   ✅ Kept    : {len(kept)}")
    print(f"   ❌ Removed : {len(removed)}")
    if removal_reasons:
        print(f"   Removal reasons:")
        for reason, count in sorted(removal_reasons.items(), key=lambda x: -x[1])[:8]:
            print(f"     • {reason}: {count}")

    return {
        "file": path.name,
        "original": len(records),
        "kept": len(kept),
        "removed": len(removed),
    }


def fix_test_script(test_script_path: str = "./test_skillforge.py"):
    """Fix max_tokens from 600 → 800 in test_skillforge.py."""
    path = Path(test_script_path)
    if not path.exists():
        print(f"\n⚠️  test_skillforge.py not found at {test_script_path} — skipping fix")
        return

    content = path.read_text()
    if "max_tokens=600" in content:
        content = content.replace("max_tokens=600", "max_tokens=800")
        path.write_text(content)
        print(f"\n✅ Fixed test_skillforge.py: max_tokens 600 → 800")
    elif "max_tokens=800" in content:
        print(f"\n✅ test_skillforge.py already has max_tokens=800")
    else:
        print(f"\n⚠️  Could not find max_tokens in test_skillforge.py — check manually")


def main():
    print("=" * 55)
    print("  SkillVault Data Cleaner")
    print("=" * 55)

    data_dir = Path("./skillvault_mlx")
    if not data_dir.exists():
        print(f"\n❌ Directory not found: {data_dir}")
        print("   Run from your SkillForge project root.")
        return

    # Clean both splits
    stats = []
    for filename in ["train.jsonl", "valid.jsonl"]:
        filepath = data_dir / filename
        result = clean_file(str(filepath))
        if result:
            stats.append(result)

    # Summary
    if stats:
        total_orig    = sum(s["original"] for s in stats)
        total_kept    = sum(s["kept"]     for s in stats)
        total_removed = sum(s["removed"]  for s in stats)
        pct_kept = (total_kept / total_orig * 100) if total_orig else 0

        print(f"\n{'=' * 55}")
        print(f"  Summary")
        print(f"{'=' * 55}")
        print(f"  Original  : {total_orig:,} records")
        print(f"  Kept      : {total_kept:,} records ({pct_kept:.1f}%)")
        print(f"  Removed   : {total_removed:,} records")

        if pct_kept < 50:
            print(f"\n  ⚠️  Over 50% removed — dataset may be too small.")
            print(f"     Consider loosening MIN_WORD_COUNT or noise keywords.")
        elif pct_kept > 80:
            print(f"\n  ✅ Good retention rate — clean dataset ready.")

    # Fix test script
    fix_test_script("./test_skillforge.py")

    print(f"\n📌 Next Steps:")
    print(f"  1. Resume training from cleaned data:")
    print(f"     python3 -m mlx_lm lora \\")
    print(f"       --model ./llama-3.2-3b \\")
    print(f"       --train \\")
    print(f"       --data $(pwd)/skillvault_mlx \\")
    print(f"       --resume-adapter-file ./skillforge-adapters/adapters.safetensors \\")
    print(f"       --iters 2000 \\")
    print(f"       --batch-size 1 \\")
    print(f"       --num-layers 8 \\")
    print(f"       --learning-rate 1e-4 \\")
    print(f"       --steps-per-report 10 \\")
    print(f"       --steps-per-eval 100 \\")
    print(f"       --val-batches 10 \\")
    print(f"       --save-every 100 \\")
    print(f"       --max-seq-length 1024 \\")
    print(f"       --grad-accumulation-steps 4 \\")
    print(f"       --adapter-path ./skillforge-adapters-v2")
    print(f"\n  2. Test v2 model:")
    print(f"     python3 test_skillforge.py")


if __name__ == "__main__":
    main()