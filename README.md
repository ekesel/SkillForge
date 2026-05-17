---
language: [en]
license: llama3.2
base_model: meta-llama/Llama-3.2-3B-Instruct
tags:
  - skill-files
  - agent-skills
  - claude-code
  - llm-instructions
  - fine-tuned
  - mlx
---

# SkillForge — LLaMA 3.2 3B Fine-tuned on SkillVault

A domain-specific model trained to generate professional AI agent 
skill files (SKILL.md) for Claude Code, Cursor, Gemini CLI, and Codex.

## What It Does

Given a tool or task description, SkillForge generates a complete,
structured SKILL.md file with proper YAML frontmatter, trigger 
conditions, prerequisites, and practical code examples.

## Training

- **Base model:** meta-llama/Llama-3.2-3B-Instruct
- **Method:** LoRA fine-tuning via MLX on Apple M4
- **Dataset:** SkillVault — 1,113 curated SKILL.md files from
  trusted repos (Anthropic, LambdaTest, and community collections)
  + 10 hand-crafted synthetic examples
- **Iterations:** 2,000 (1,000 @ 2e-4 LR + 1,000 @ 1e-4 LR)
- **Final train loss:** 0.656 | **Val loss:** 1.312

## Usage

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="ekesel/skillforge-llama-3.2-3b")

prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are SkillForge, an expert AI that generates professional agent skill files (SKILL.md).<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Generate a SKILL.md for PostgreSQL database management.

Description: Best practices for queries, schemas, and debugging
Allowed tools: Bash, Read, Write<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

result = pipe(prompt, max_new_tokens=600, temperature=0.7)
print(result[0]['generated_text'])
```

## Dataset

The training dataset (SkillVault) is available at:
`ekesel/skillvault`
