"""
compare_one.py
==============
Run one prompt through both base and SkillForge model.
Saves output to compare_output.md

Usage:
    python3 compare_one.py
"""

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_repetition_penalty
from datetime import datetime
import gc

# ── CONFIG ──────────────────────────────────
MODEL_PATH   = "./llama-3.2-3b"
ADAPTER_PATH = "./skillforge-adapters-v3"

SYSTEM_PROMPT = (
    "You are SkillForge, an expert AI that generates professional "
    "agent skill files (SKILL.md). You produce complete, accurate, "
    "and well-structured SKILL.md files with proper YAML frontmatter, "
    "clear instructions, trigger conditions, and examples."
)

# ── YOUR PROMPT ─────────────────────────────
INSTRUCTION = "Generate a SKILL.md for working with PostgreSQL databases."
USER_INPUT  = "Description: Best practices for writing queries, managing schemas, and debugging PostgreSQL\nAllowed tools: Bash, Read, Write"
# ────────────────────────────────────────────


def run(model, tokenizer):
    user_msg = f"{INSTRUCTION}\n\n{USER_INPUT}" if USER_INPUT.strip() else INSTRUCTION
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    sampler     = make_sampler(temp=0.7)
    rep_penalty = make_repetition_penalty(penalty=1.2, context_size=20)
    return generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=800,
        sampler=sampler,
        logits_processors=[rep_penalty],
        verbose=False,
    )


def main():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Base model ──
    print("📦 Loading base model...")
    base_model, base_tok = load(MODEL_PATH)
    print("🧪 Running prompt through base model...")
    base_out = run(base_model, base_tok)
    del base_model, base_tok
    gc.collect()
    print("✅ Base model done\n")

    # ── SkillForge model ──
    print("🔥 Loading SkillForge model...")
    ft_model, ft_tok = load(MODEL_PATH, adapter_path=ADAPTER_PATH)
    print("🧪 Running prompt through SkillForge model...")
    ft_out = run(ft_model, ft_tok)
    del ft_model, ft_tok
    gc.collect()
    print("✅ SkillForge model done\n")

    # ── Build markdown ──
    md = f"""# Base vs SkillForge — Single Prompt Comparison
**Generated:** {now}

**Prompt:** {INSTRUCTION}  
**Input:** {USER_INPUT}

---

## Base Model — LLaMA 3.2 3B (no fine-tuning)

```
{base_out.strip()}
```

---

## SkillForge v2 — Fine-tuned on SkillVault

```
{ft_out.strip()}
```

---

## What to compare
- Does SkillForge use `name:` (not `title:`)?
- Does SkillForge have `allowed-tools:` in frontmatter?
- Does SkillForge follow SKILL.md structure (not just prose)?
- Which output would you actually use in a real project?
"""

    with open("compare_output.md", "w") as f:
        f.write(md)

    print("💾 Saved → compare_output.md")
    print("\n--- BASE MODEL ---")
    print(base_out.strip()[:400])
    print("\n--- SKILLFORGE ---")
    print(ft_out.strip()[:400])


if __name__ == "__main__":
    main()