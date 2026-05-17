"""
test_skillforge.py
==================
Runs the same 5 prompts through:
  1. Base LLaMA 3.2 3B (no adapters)
  2. SkillForge fine-tuned model (with adapters)

Saves results to:
  - base_model_output.md
  - skillforge_output.md

Usage:
    python3 test_skillforge.py
"""

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_repetition_penalty
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH   = "./llama-3.2-3b"
ADAPTER_PATH = "./skillforge-adapters-v2"

SYSTEM_PROMPT = (
    "You are SkillForge, an expert AI that generates professional "
    "agent skill files (SKILL.md). You deeply understand how AI coding "
    "agents like Claude Code, Cursor, Gemini CLI, and Codex use skill "
    "files to perform specialized tasks. You produce complete, accurate, "
    "and well-structured SKILL.md files with proper YAML frontmatter, "
    "clear instructions, trigger conditions, and examples."
)

# ─────────────────────────────────────────────
# TEST PROMPTS
# 5 diverse prompts — covers different skill types
# ─────────────────────────────────────────────
TEST_PROMPTS = [
    {
        "title": "Django REST API",
        "instruction": "Generate a SKILL.md for Django REST API development.",
        "input": "Description: Best practices for building and debugging Django REST Framework APIs\nAllowed tools: Bash, Read, Write",
    },
    {
        "title": "Redis Caching",
        "instruction": "Generate a SKILL.md for Redis caching in Python applications.",
        "input": "Description: Implementing and debugging Redis cache operations with connection pooling\nAllowed tools: Bash, Read",
    },
    {
        "title": "Celery Task Queue",
        "instruction": "Generate a SKILL.md for Celery task queue management.",
        "input": "Description: Creating, monitoring, and debugging Celery tasks with Redis broker\nAllowed tools: Bash, Read, Write",
    },
    {
        "title": "Describe a Skill",
        "instruction": "What does the neon-postgres skill do and when should it be used?",
        "input": "",
    },
    {
        "title": "FastAPI Microservice",
        "instruction": "Generate a SKILL.md for building FastAPI microservices.",
        "input": "Description: Creating FastAPI endpoints, dependency injection, and async handlers\nAllowed tools: Bash, Read, Write, Edit",
    },
]


# ─────────────────────────────────────────────
# GENERATE
# ─────────────────────────────────────────────
def run_prompt(model, tokenizer, instruction, user_input=""):
    """Format prompt using LLaMA chat template and generate response."""
    user_msg = f"{instruction}\n\n{user_input}" if user_input.strip() else instruction

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    sampler = make_sampler(temp=0.7)
    rep_penalty = make_repetition_penalty(penalty=1.2, context_size=20)

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=800,
        sampler=sampler,
        logits_processors=[rep_penalty],
        verbose=False,
    )
    return response


def build_md(label, model_desc, results):
    """Build a markdown report from test results."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# {label}",
        f"**Model:** {model_desc}  ",
        f"**Generated:** {now}  ",
        f"**Prompts tested:** {len(results)}",
        "",
        "---",
        "",
    ]

    for i, r in enumerate(results, 1):
        lines += [
            f"## Test {i} — {r['title']}",
            "",
            f"**Instruction:** {r['instruction']}  ",
        ]
        if r["input"]:
            lines += [f"**Input:** {r['input']}  "]
        lines += [
            "",
            "**Output:**",
            "",
            "```",
            r["output"].strip(),
            "```",
            "",
            "---",
            "",
        ]

    return "\n".join(lines)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  SkillForge — Base vs Fine-tuned Comparison")
    print("=" * 55)

    # ── Run BASE MODEL (no adapters) ──
    print("\n📦 Loading BASE model (no adapters)...")
    base_model, base_tokenizer = load(MODEL_PATH)

    print("🧪 Running 5 prompts through BASE model...")
    base_results = []
    for i, p in enumerate(TEST_PROMPTS, 1):
        print(f"   [{i}/5] {p['title']}...", end=" ", flush=True)
        output = run_prompt(base_model, base_tokenizer, p["instruction"], p["input"])
        base_results.append({**p, "output": output})
        print("✅")

    # Save base output
    base_md = build_md(
        "Base Model Output — LLaMA 3.2 3B (No Fine-tuning)",
        "meta-llama/Llama-3.2-3B-Instruct (base, no adapters)",
        base_results,
    )
    with open("base_model_output.md", "w") as f:
        f.write(base_md)
    print(f"\n💾 Saved → base_model_output.md")

    # Free base model memory before loading fine-tuned
    del base_model, base_tokenizer
    import gc
    gc.collect()

    # ── Run FINE-TUNED MODEL (with adapters) ──
    print("\n🔥 Loading SKILLFORGE model (with adapters)...")
    ft_model, ft_tokenizer = load(MODEL_PATH, adapter_path=ADAPTER_PATH)

    print("🧪 Running 5 prompts through SKILLFORGE model...")
    ft_results = []
    for i, p in enumerate(TEST_PROMPTS, 1):
        print(f"   [{i}/5] {p['title']}...", end=" ", flush=True)
        output = run_prompt(ft_model, ft_tokenizer, p["instruction"], p["input"])
        ft_results.append({**p, "output": output})
        print("✅")

    # Save fine-tuned output
    ft_md = build_md(
        "SkillForge Output — LLaMA 3.2 3B Fine-tuned on SkillVault",
        f"SkillForge v1.0 — fine-tuned with LoRA adapters from {ADAPTER_PATH}",
        ft_results,
    )
    with open("skillforge_output.md", "w") as f:
        f.write(ft_md)
    print(f"💾 Saved → skillforge_output.md")

    # ── Summary ──
    print("\n" + "=" * 55)
    print("  ✅  Done!")
    print("=" * 55)
    print("  Open these files and compare side by side:")
    print("  📄  base_model_output.md")
    print("  📄  skillforge_output.md")
    print()
    print("  What to look for:")
    print("  ✅  SkillForge should have YAML frontmatter")
    print("  ✅  SkillForge should use correct ## sections")
    print("  ✅  SkillForge should have TRIGGER conditions")
    print("  ✅  Base model will likely ignore the format")


if __name__ == "__main__":
    main()