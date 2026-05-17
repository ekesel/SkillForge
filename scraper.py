"""
SkillVault Scraper
==================
Scrapes SKILL.md files from all official and community skill repos
listed in VoltAgent/awesome-agent-skills and related collections.

Outputs a HuggingFace-ready dataset of instruction → skill pairs.

Usage:
    pip install requests PyYAML datasets tqdm
    python skillvault_scraper.py --token YOUR_GITHUB_TOKEN

Get a free GitHub token at: https://github.com/settings/tokens
(No special scopes needed — public repo access is default)
"""

import os
import re
import json
import time
import argparse
import requests
import yaml
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset
from datetime import datetime

# ─────────────────────────────────────────────
# KNOWN SKILL REPOS
# Repos confirmed to contain SKILL.md files.
# Scraper also discovers more from the awesome-agent-skills README.
# ─────────────────────────────────────────────
KNOWN_SKILL_REPOS = [
    {"repo": "anthropics/skills",                    "skills_path": "skills"},
    {"repo": "vercel-labs/skills",                   "skills_path": "skills"},
    {"repo": "ComposioHQ/awesome-claude-skills",     "skills_path": None},
    {"repo": "hesreallyhim/awesome-claude-code",     "skills_path": None},
    {"repo": "heilcheng/awesome-agent-skills",       "skills_path": None},
    {"repo": "neondatabase/agent-skills",            "skills_path": None},
    {"repo": "clickhouse/agent-skills",              "skills_path": None},
    {"repo": "tinybirdco/agent-skills",              "skills_path": None},
    {"repo": "supabase/agent-skills",                "skills_path": None},
    {"repo": "sanity-io/agent-skills",               "skills_path": None},
    {"repo": "cloudflare/agent-skills",              "skills_path": None},
    {"repo": "stripe/agent-skills",                  "skills_path": None},
    {"repo": "sentry-io/agent-skills",               "skills_path": None},
    {"repo": "huggingface/agent-skills",             "skills_path": None},
    {"repo": "figma/agent-skills",                   "skills_path": None},
]


# ─────────────────────────────────────────────
# GITHUB API CLIENT
# ─────────────────────────────────────────────
class GitHubClient:
    BASE = "https://api.github.com"

    def __init__(self, token: str):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "SkillVault-Scraper/1.0",
        })
        self.request_count = 0

    def _get(self, url, params=None):
        """GET with rate limit handling and retries."""
        for attempt in range(3):
            try:
                r = self.session.get(url, params=params, timeout=15)
                self.request_count += 1

                if r.status_code == 200:
                    return r.json()
                elif r.status_code == 403:
                    reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
                    wait = max(reset - time.time(), 1)
                    print(f"\n⏳ Rate limited — waiting {wait:.0f}s...")
                    time.sleep(wait + 2)
                    continue
                elif r.status_code in (404, 422):
                    return None
                else:
                    print(f"  ⚠️  HTTP {r.status_code}: {url}")
                    return None
            except requests.exceptions.Timeout:
                print(f"  ⏱️  Timeout attempt {attempt+1}/3")
                time.sleep(2 ** attempt)
        return None

    def get_file_content(self, repo, path):
        """Fetch raw content of a file."""
        import base64
        data = self._get(f"{self.BASE}/repos/{repo}/contents/{path}")
        if data and isinstance(data, dict) and data.get("type") == "file":
            try:
                return base64.b64decode(data["content"]).decode("utf-8")
            except Exception:
                return None
        return None

    def list_skill_files(self, repo, skills_path=None):
        """Find all SKILL.md files using the Git Trees API (fast, one request)."""
        # Get default branch
        repo_data = self._get(f"{self.BASE}/repos/{repo}")
        if not repo_data:
            return []

        branch = repo_data.get("default_branch", "main")
        tree_data = self._get(f"{self.BASE}/repos/{repo}/git/trees/{branch}?recursive=1")
        if not tree_data or "tree" not in tree_data:
            return []

        skill_files = []
        for item in tree_data["tree"]:
            if item["type"] != "blob":
                continue
            path = item["path"]
            if not path.endswith("SKILL.md"):
                continue
            if skills_path and not path.startswith(skills_path):
                continue
            skill_files.append(path)

        return skill_files

    def get_rate_limit(self):
        data = self._get(f"{self.BASE}/rate_limit")
        if data:
            core = data["resources"]["core"]
            return {
                "remaining": core["remaining"],
                "limit": core["limit"],
                "reset_in": max(0, core["reset"] - time.time()),
            }
        return {}


# ─────────────────────────────────────────────
# README PARSER
# Discovers additional repos from awesome-agent-skills README
# ─────────────────────────────────────────────
def parse_readme_for_repos(client):
    """Extract GitHub repo references from the awesome-agent-skills README."""
    print("\n📖 Parsing awesome-agent-skills README...")
    content = client.get_file_content("VoltAgent/awesome-agent-skills", "README.md")
    if not content:
        print("  ⚠️  Could not fetch README — using known repos only")
        return []

    # Extract all github.com/org/repo URLs
    github_urls = re.findall(
        r'https://github\.com/([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)',
        content
    )

    # Deduplicate and filter
    repos = []
    seen = set()
    for ref in github_urls:
        parts = ref.split("/")
        if len(parts) == 2 and ref not in seen:
            # Skip obvious non-code repos
            if not any(x in ref.lower() for x in ["shields.io", "badge", "img."]):
                repos.append(ref)
                seen.add(ref)

    print(f"  ✅ Found {len(repos)} repo references")
    return repos


# ─────────────────────────────────────────────
# SKILL PARSER
# ─────────────────────────────────────────────
def parse_skill_file(content, repo, path):
    """Parse a SKILL.md into a structured dict."""
    if not content or len(content.strip()) < 50:
        return None

    frontmatter = {}
    body = content

    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                frontmatter = yaml.safe_load(parts[1]) or {}
                body = parts[2].strip()
            except yaml.YAMLError:
                body = content

    if len(body.strip()) < 30:
        return None

    name = (
        frontmatter.get("name") or
        frontmatter.get("title") or
        Path(path).parent.name or
        "unknown"
    )
    description = str(frontmatter.get("description") or frontmatter.get("desc") or "")
    tools = frontmatter.get("allowed-tools") or frontmatter.get("tools") or []

    return {
        "name": str(name),
        "description": description,
        "allowed_tools": tools if isinstance(tools, list) else [tools],
        "content": body,
        "raw": content,
        "repo": repo,
        "path": path,
        "source_url": f"https://github.com/{repo}/blob/main/{path}",
        "char_count": len(body),
        "word_count": len(body.split()),
    }


# ─────────────────────────────────────────────
# TRAINING PAIR GENERATOR
# ─────────────────────────────────────────────
def generate_training_pairs(skill):
    """Generate instruction → output training pairs from a skill."""
    pairs = []
    name = skill["name"]
    description = skill["description"]
    content = skill["content"]
    tools = skill["allowed_tools"]
    tool_str = ", ".join(tools) if tools else "standard file operations"

    # Pair 1: Generate SKILL.md from description
    if description:
        pairs.append({
            "instruction": f'Generate a complete SKILL.md file for an AI agent skill called "{name}".',
            "input": f"Description: {description}\nAllowed tools: {tool_str}",
            "output": skill["raw"],
            "pair_type": "generate_from_description",
            "skill_name": name,
            "source": skill["source_url"],
        })

    # Pair 2: What does this skill do?
    if description:
        pairs.append({
            "instruction": f'What does the "{name}" AI agent skill do? When should it be activated?',
            "input": "",
            "output": description,
            "pair_type": "describe_skill",
            "skill_name": name,
            "source": skill["source_url"],
        })

    # Pair 3: Generate from use case
    if len(content) > 200:
        pairs.append({
            "instruction": "Write a SKILL.md file for an AI coding agent based on this use case.",
            "input": f"Use case: {description or name}\nTools available: {tool_str}",
            "output": skill["raw"],
            "pair_type": "generate_from_usecase",
            "skill_name": name,
            "source": skill["source_url"],
        })

    # Pair 4: Extract trigger conditions (if present in content)
    trigger = re.search(
        r'TRIGGER[^\n]*\n(.*?)(?:SKIP|##|\Z)',
        content, re.DOTALL | re.IGNORECASE
    )
    if trigger and len(trigger.group(1).strip()) > 20:
        pairs.append({
            "instruction": f'List the trigger conditions for the "{name}" AI agent skill.',
            "input": "",
            "output": trigger.group(1).strip(),
            "pair_type": "trigger_conditions",
            "skill_name": name,
            "source": skill["source_url"],
        })

    # Pair 5: Full skill reconstruction (instruction tuning style)
    pairs.append({
        "instruction": "You are a SkillForge AI. Generate a professional SKILL.md file.",
        "input": f"Skill name: {name}\nPurpose: {description or 'See content'}\nTools: {tool_str}",
        "output": skill["raw"],
        "pair_type": "skillforge_generate",
        "skill_name": name,
        "source": skill["source_url"],
    })

    return pairs


# ─────────────────────────────────────────────
# MAIN SCRAPER
# ─────────────────────────────────────────────
def scrape_all_skills(token, output_dir="./skillvault_data"):
    client = GitHubClient(token)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check rate limit status
    rl = client.get_rate_limit()
    if rl:
        print(f"\n📊 GitHub API: {rl['remaining']}/{rl['limit']} requests remaining")
        if rl["remaining"] < 100:
            print(f"⚠️  Low rate limit — resets in {rl['reset_in']:.0f}s")

    # Discover repos
    extra_repos = parse_readme_for_repos(client)

    all_repos = {r["repo"]: r.get("skills_path") for r in KNOWN_SKILL_REPOS}
    for repo in extra_repos:
        if repo not in all_repos:
            all_repos[repo] = None

    print(f"\n🔍 Scanning {len(all_repos)} repos for SKILL.md files...")

    # Load checkpoint
    all_skills = []
    seen_paths = set()
    checkpoint_file = output_path / "checkpoint.jsonl"

    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            for line in f:
                try:
                    skill = json.loads(line)
                    all_skills.append(skill)
                    seen_paths.add(f"{skill['repo']}/{skill['path']}")
                except:
                    pass
        if all_skills:
            print(f"  ♻️  Checkpoint: {len(all_skills)} skills already collected")

    # Scrape each repo
    with open(checkpoint_file, "a") as checkpoint:
        for repo, skills_path in tqdm(all_repos.items(), desc="Scanning repos"):
            try:
                skill_files = client.list_skill_files(repo, skills_path)
                if not skill_files:
                    continue

                tqdm.write(f"  📂 {repo}: {len(skill_files)} SKILL.md found")

                for file_path in skill_files:
                    uid = f"{repo}/{file_path}"
                    if uid in seen_paths:
                        continue

                    content = client.get_file_content(repo, file_path)
                    if not content:
                        continue

                    skill = parse_skill_file(content, repo, file_path)
                    if not skill:
                        continue

                    all_skills.append(skill)
                    seen_paths.add(uid)
                    checkpoint.write(json.dumps(skill, ensure_ascii=False) + "\n")
                    checkpoint.flush()
                    time.sleep(0.1)  # Polite delay

            except Exception as e:
                tqdm.write(f"  ❌ {repo}: {e}")

    print(f"\n✅ Total SKILL.md files collected: {len(all_skills)}")
    return all_skills


# ─────────────────────────────────────────────
# DATASET BUILDER
# ─────────────────────────────────────────────
def build_dataset(skills, output_dir="./skillvault_data"):
    output_path = Path(output_dir)
    print(f"\n🏗️  Building dataset from {len(skills)} skills...")

    # Quality filter
    quality = [s for s in skills if s.get("word_count", 0) >= 100 and s.get("name") not in ("", "unknown", None)]
    print(f"  📋 Quality filter: {len(quality)}/{len(skills)} passed")

    # Save raw skills
    raw_path = output_path / "skills_raw.jsonl"
    with open(raw_path, "w") as f:
        for s in quality:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  💾 Raw skills → {raw_path}")

    # Generate training pairs
    all_pairs = []
    for skill in quality:
        all_pairs.extend(generate_training_pairs(skill))
    print(f"  🎯 Training pairs generated: {len(all_pairs)}")

    # Save pairs
    pairs_path = output_path / "training_pairs.jsonl"
    with open(pairs_path, "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  💾 Training pairs → {pairs_path}")

    # HuggingFace Dataset with train/val split
    hf_ds = Dataset.from_list(all_pairs)
    split = hf_ds.train_test_split(test_size=0.1, seed=42)
    hf_path = output_path / "hf_dataset"
    split.save_to_disk(str(hf_path))
    print(f"  🤗 HuggingFace dataset → {hf_path}")
    print(f"      Train: {len(split['train'])} | Val: {len(split['test'])}")

    # Stats
    repos = sorted(set(s["repo"] for s in quality))
    pair_types = {}
    for p in all_pairs:
        t = p.get("pair_type", "other")
        pair_types[t] = pair_types.get(t, 0) + 1

    stats = {
        "total_skills": len(quality),
        "total_pairs": len(all_pairs),
        "unique_repos": len(repos),
        "repos": repos,
        "pair_types": pair_types,
        "avg_words": sum(s.get("word_count", 0) for s in quality) // max(len(quality), 1),
        "created_at": datetime.now().isoformat(),
    }

    # Save stats
    stats_path = output_path / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  📊 Stats → {stats_path}")

    # Write dataset card
    card = f"""---
language: [en]
license: mit
tags: [agent-skills, skill-files, claude-code, llm-instructions, ai-tools]
---

# SkillVault Dataset

The world's first assembled dataset of AI agent SKILL.md files.

Scraped from **{stats['unique_repos']} GitHub repos** across Anthropic, Vercel,
Cloudflare, Stripe, Neon, ClickHouse, and the broader community.

## Stats
- **Skills collected:** {stats['total_skills']}
- **Training pairs:** {stats['total_pairs']}
- **Source repos:** {stats['unique_repos']}
- **Avg skill length:** {stats['avg_words']} words

## Training Pair Types
{chr(10).join(f"- `{k}`: {v}" for k, v in pair_types.items())}

## Usage
```python
from datasets import load_from_disk
ds = load_from_disk("./skillvault_data/hf_dataset")
print(ds["train"][0])
```
"""
    (output_path / "README.md").write_text(card)

    return split, stats


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SkillVault — AI Agent Skills Dataset Builder")
    parser.add_argument("--token", "-t", required=True, help="GitHub token (github.com/settings/tokens)")
    parser.add_argument("--output", "-o", default="./skillvault_data", help="Output directory")
    parser.add_argument("--hf-repo", default=None, help="HuggingFace repo ID to push to (e.g. yourname/skillvault)")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token")
    args = parser.parse_args()

    print("=" * 55)
    print("  SkillVault Scraper — AI Agent Skills Dataset")
    print("=" * 55)

    skills = scrape_all_skills(args.token, args.output)

    if not skills:
        print("\n❌ No skills found. Check your GitHub token.")
        return

    dataset, stats = build_dataset(skills, args.output)

    print("\n" + "=" * 55)
    print("  ✅  Done!")
    print("=" * 55)
    print(f"  Skills:    {stats['total_skills']}")
    print(f"  Pairs:     {stats['total_pairs']}")
    print(f"  Repos:     {stats['unique_repos']}")
    print(f"  Output:    {args.output}/")

    if args.hf_repo and args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)
        dataset.push_to_hub(args.hf_repo)
        print(f"\n🚀 Pushed → huggingface.co/datasets/{args.hf_repo}")

    print("\n📌 Next Steps:")
    print("  1. Review  → skillvault_data/skills_raw.jsonl")
    print("  2. Inspect → skillvault_data/training_pairs.jsonl")
    print("  3. Train   → skillvault_data/hf_dataset/")
    print("  4. Upload  → huggingface-cli upload yourname/skillvault skillvault_data/hf_dataset")


if __name__ == "__main__":
    main()