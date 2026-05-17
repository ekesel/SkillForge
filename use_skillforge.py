# use_skillforge.py
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_repetition_penalty

model, tokenizer = load(
    "./llama-3.2-3b",
    adapter_path="./skillforge-adapters-v3"
)

SYSTEM = "You are SkillForge, an expert AI that generates professional agent skill files (SKILL.md)."

def generate_skill(tool_name, description, allowed_tools):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": f"Generate a SKILL.md for {tool_name}.\n\nDescription: {description}\nAllowed tools: {allowed_tools}"},
    ]
    prompt   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sampler  = make_sampler(temp=0.7)
    rep_pen  = make_repetition_penalty(penalty=1.2, context_size=20)

    return generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=800,
        sampler=sampler,
        logits_processors=[rep_pen],
        verbose=False,
    )

# ── Use it ──
skill = generate_skill(
    tool_name     = "Django REST Framework",
    description   = "Best practices for building DRF APIs",
    allowed_tools = "Bash, Read, Write"
)
print(skill)