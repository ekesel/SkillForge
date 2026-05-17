"""
rebuild_dataset.py
==================
Rebuilds the SkillForge training dataset from scratch.

Strategy:
  - Whitelist only trusted, high-quality repos
  - Filter skills with > 300 words (substance check)
  - ONE clean training pair per skill
  - Add 30 hand-crafted synthetic examples
  - Retrain from scratch

Usage:
    python3 rebuild_dataset.py
"""

import json
import random
from pathlib import Path

# ─────────────────────────────────────────────
# TRUSTED REPOS WHITELIST
# Only these repos produce clean, professional skills
# ─────────────────────────────────────────────
TRUSTED_REPOS = {
    "anthropics/skills",                    # Official Anthropic skills — highest quality
    "LambdaTest/agent-skills",              # Professional testing company
    "phuryn/pm-skills",                     # Curated product management skills
    "NeoLabHQ/context-engineering-kit",     # Context engineering
    "deanpeters/Product-Manager-Skills",    # Curated PM skills
    "lawvable/awesome-legal-skills",        # Legal domain skills
    "coreyhaines31/marketingskills",        # Marketing skills
    "bitwize-music-studio/claude-ai-music-skills",  # Music domain
    "alinaqi/claude-bootstrap",             # Bootstrap patterns
    "mukul975/Anthropic-Cybersecurity-Skills",  # Cybersecurity (keep, filter by quality)
    "hesreallyhim/awesome-claude-code",     # Community collection
}

# Minimum word count for a skill to be included
MIN_WORDS = 300

# System prompt used during training
SYSTEM_PROMPT = (
    "You are SkillForge, an expert AI that generates professional agent skill "
    "files (SKILL.md). You deeply understand how AI coding agents like Claude Code, "
    "Cursor, Gemini CLI, and Codex use skill files to perform specialized tasks. "
    "You produce complete, accurate, and well-structured SKILL.md files with proper "
    "YAML frontmatter (name, description, allowed-tools), clear instructions, "
    "trigger conditions, and practical examples."
)


# ─────────────────────────────────────────────
# SYNTHETIC EXAMPLES
# 30 hand-crafted perfect SKILL.md examples
# covering common developer tools
# ─────────────────────────────────────────────
SYNTHETIC_SKILLS = [
    {
        "name": "django-rest-api",
        "description": "Best practices for Django REST Framework API development. Use when building, debugging, or reviewing DRF endpoints, serializers, viewsets, and permissions.",
        "allowed_tools": ["Bash", "Read", "Write", "Edit"],
        "content": """---
name: django-rest-api
description: Best practices for Django REST Framework API development. Use when building, debugging, or reviewing DRF endpoints, serializers, viewsets, and permissions.
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
---

# Django REST Framework API Development

## TRIGGER
Use this skill when the user:
- Asks to build or modify Django REST API endpoints
- Needs to debug DRF serializers, viewsets, or permissions
- Wants to add authentication to a Django API
- Asks about DRF best practices

## SKIP
Skip this skill for non-DRF Django views or plain Django ORM questions.

## Setup

```bash
pip install djangorestframework
# Add 'rest_framework' to INSTALLED_APPS in settings.py
```

## Core Patterns

### Serializer
```python
from rest_framework import serializers
from .models import Transaction

class TransactionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Transaction
        fields = ['id', 'amount', 'status', 'created_at']
        read_only_fields = ['id', 'created_at']
```

### ViewSet
```python
from rest_framework import viewsets, permissions
from rest_framework.decorators import action
from rest_framework.response import Response

class TransactionViewSet(viewsets.ModelViewSet):
    serializer_class = TransactionSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Transaction.objects.filter(user=self.request.user)

    @action(detail=True, methods=['post'])
    def approve(self, request, pk=None):
        transaction = self.get_object()
        transaction.status = 'approved'
        transaction.save()
        return Response({'status': 'approved'})
```

### Router
```python
from rest_framework.routers import DefaultRouter
router = DefaultRouter()
router.register(r'transactions', TransactionViewSet, basename='transaction')
urlpatterns = router.urls
```

## Authentication
```python
# settings.py
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,
}
```

## Debugging
```bash
# Run server with verbose logging
python manage.py runserver --verbosity=2

# Test endpoint directly
curl -H "Authorization: Bearer <token>" http://localhost:8000/api/transactions/
```
"""
    },
    {
        "name": "redis-cache-python",
        "description": "Redis caching patterns for Python applications. Use when implementing cache-aside, write-through, or session caching with the redis-py library.",
        "allowed_tools": ["Bash", "Read", "Write"],
        "content": """---
name: redis-cache-python
description: Redis caching patterns for Python applications. Use when implementing cache-aside, write-through, or session caching with the redis-py library.
allowed-tools:
  - Bash
  - Read
  - Write
---

# Redis Caching for Python

## TRIGGER
Use this skill when:
- Adding caching to a Python/Django/FastAPI application
- Debugging cache misses or stale data
- Implementing session storage with Redis
- Setting up connection pooling for Redis

## Setup

```bash
pip install redis
```

## Connection with Pooling
```python
import redis

pool = redis.ConnectionPool(
    host='localhost',
    port=6379,
    db=0,
    max_connections=20,
    decode_responses=True,
)
client = redis.Redis(connection_pool=pool)
```

## Cache-Aside Pattern
```python
import json
from functools import wraps

def cache(key_prefix, ttl=300):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{key_prefix}:{args[0]}"
            cached = client.get(cache_key)
            if cached:
                return json.loads(cached)
            result = func(*args, **kwargs)
            client.setex(cache_key, ttl, json.dumps(result))
            return result
        return wrapper
    return decorator

@cache("user", ttl=600)
def get_user(user_id):
    return User.objects.get(id=user_id).to_dict()
```

## Common Operations
```python
# Set with expiry
client.setex("session:abc123", 3600, json.dumps(session_data))

# Atomic increment (counters, rate limiting)
client.incr("api:calls:user:42")
client.expire("api:calls:user:42", 60)

# Hash (store object fields)
client.hset("user:42", mapping={"name": "Ekaansh", "role": "admin"})
client.hget("user:42", "name")

# Delete pattern
for key in client.scan_iter("session:*"):
    client.delete(key)
```

## Debugging
```bash
redis-cli monitor          # Watch all Redis commands in real time
redis-cli info memory      # Check memory usage
redis-cli --bigkeys        # Find largest keys
redis-cli ttl "key:name"   # Check remaining TTL
```
"""
    },
    {
        "name": "celery-task-queue",
        "description": "Celery task queue management with Redis broker. Use when creating async tasks, scheduling periodic jobs, or debugging task failures.",
        "allowed_tools": ["Bash", "Read", "Write"],
        "content": """---
name: celery-task-queue
description: Celery task queue management with Redis broker. Use when creating async tasks, scheduling periodic jobs, or debugging task failures.
allowed-tools:
  - Bash
  - Read
  - Write
---

# Celery Task Queue Management

## TRIGGER
Use this skill when:
- Creating or modifying Celery tasks
- Debugging task failures or timeouts
- Setting up periodic tasks with Celery Beat
- Monitoring task queue health

## Setup

```bash
pip install celery redis
```

## Configuration
```python
# celery.py
from celery import Celery

app = Celery('myproject')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()

# settings.py
CELERY_BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_TIME_LIMIT = 30 * 60
```

## Defining Tasks
```python
from celery import shared_task
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def process_transaction(self, transaction_id):
    try:
        transaction = Transaction.objects.get(id=transaction_id)
        result = run_risk_scoring(transaction)
        return {'transaction_id': transaction_id, 'score': result}
    except Exception as exc:
        logger.error(f"Task failed for {transaction_id}: {exc}")
        raise self.retry(exc=exc)
```

## Running Workers
```bash
# Start worker
celery -A myproject worker --loglevel=info --concurrency=4

# Start beat scheduler
celery -A myproject beat --loglevel=info

# Monitor tasks
celery -A myproject flower  # Web UI at localhost:5555
```

## Debugging
```bash
# Inspect active tasks
celery -A myproject inspect active

# Check registered tasks
celery -A myproject inspect registered

# Purge all tasks from queue
celery -A myproject purge

# View task result
from celery.result import AsyncResult
result = AsyncResult('task-id-here')
print(result.status, result.result)
```
"""
    },
    {
        "name": "fastapi-microservice",
        "description": "FastAPI microservice development. Use when building async REST APIs, adding dependency injection, background tasks, or Pydantic validation.",
        "allowed_tools": ["Bash", "Read", "Write", "Edit"],
        "content": """---
name: fastapi-microservice
description: FastAPI microservice development. Use when building async REST APIs, adding dependency injection, background tasks, or Pydantic validation.
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
---

# FastAPI Microservice Development

## TRIGGER
Use this skill when:
- Building new FastAPI endpoints
- Adding Pydantic models for request/response validation
- Implementing dependency injection
- Adding background tasks or middleware

## Setup

```bash
pip install fastapi uvicorn pydantic
```

## Core Structure
```python
from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Risk Scoring API", version="1.0.0")

class TransactionRequest(BaseModel):
    amount: float
    merchant_id: str
    user_id: int
    currency: str = "INR"

class TransactionResponse(BaseModel):
    transaction_id: str
    risk_score: float
    status: str
    flagged: bool
```

## Dependency Injection
```python
from fastapi import Header

async def verify_tenant(x_tenant: str = Header(...)):
    if x_tenant not in ALLOWED_TENANTS:
        raise HTTPException(status_code=403, detail="Invalid tenant")
    return x_tenant

@app.post("/score", response_model=TransactionResponse)
async def score_transaction(
    request: TransactionRequest,
    tenant: str = Depends(verify_tenant),
):
    score = await calculate_risk_score(request, tenant)
    return TransactionResponse(
        transaction_id=generate_id(),
        risk_score=score,
        status="processed",
        flagged=score > 0.8,
    )
```

## Background Tasks
```python
from fastapi import BackgroundTasks

def send_alert(transaction_id: str, score: float):
    # Send async notification
    notify_compliance_team(transaction_id, score)

@app.post("/score")
async def score(req: TransactionRequest, bg: BackgroundTasks):
    score = await calculate_risk_score(req)
    if score > 0.8:
        bg.add_task(send_alert, req.transaction_id, score)
    return {"score": score}
```

## Running
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8001
# Docs at http://localhost:8001/docs
```
"""
    },
    {
        "name": "postgresql-queries",
        "description": "PostgreSQL query writing, schema management, and debugging. Use when writing complex SQL queries, managing migrations, or optimizing query performance.",
        "allowed_tools": ["Bash", "Read", "Write"],
        "content": """---
name: postgresql-queries
description: PostgreSQL query writing, schema management, and debugging. Use when writing complex SQL queries, managing migrations, or optimizing query performance.
allowed-tools:
  - Bash
  - Read
  - Write
---

# PostgreSQL Queries and Schema Management

## TRIGGER
Use this skill when:
- Writing complex SQL queries with JOINs, CTEs, or window functions
- Creating or modifying database schemas
- Debugging slow queries with EXPLAIN ANALYZE
- Managing indexes for performance

## Connecting
```bash
psql -U postgres -d mydb
# Or with connection string
psql "postgresql://user:password@localhost:5432/mydb"
```

## Query Patterns

### CTE (Common Table Expression)
```sql
WITH risk_summary AS (
    SELECT
        user_id,
        COUNT(*) as transaction_count,
        AVG(amount) as avg_amount,
        MAX(risk_score) as max_risk
    FROM transactions
    WHERE created_at >= NOW() - INTERVAL '30 days'
    GROUP BY user_id
)
SELECT u.email, r.*
FROM risk_summary r
JOIN users u ON u.id = r.user_id
WHERE r.max_risk > 0.8
ORDER BY r.max_risk DESC;
```

### Window Functions
```sql
SELECT
    transaction_id,
    amount,
    user_id,
    AVG(amount) OVER (PARTITION BY user_id ORDER BY created_at
                      ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as rolling_avg,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY created_at DESC) as rn
FROM transactions;
```

### Schema Management
```sql
-- Add column safely
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS flagged BOOLEAN DEFAULT FALSE;

-- Create index concurrently (no table lock)
CREATE INDEX CONCURRENTLY idx_transactions_user_created
ON transactions (user_id, created_at DESC);

-- Check index usage
SELECT indexname, idx_scan, idx_tup_read
FROM pg_stat_user_indexes
WHERE tablename = 'transactions';
```

## Debugging
```bash
# Explain query plan
EXPLAIN ANALYZE SELECT * FROM transactions WHERE user_id = 42;

# Find slow queries
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;

# Table size
SELECT pg_size_pretty(pg_total_relation_size('transactions'));
```
"""
    },
    {
        "name": "docker-containers",
        "description": "Docker container management and Dockerfile best practices. Use when writing Dockerfiles, composing multi-service stacks, or debugging container issues.",
        "allowed_tools": ["Bash", "Read", "Write"],
        "content": """---
name: docker-containers
description: Docker container management and Dockerfile best practices. Use when writing Dockerfiles, composing multi-service stacks, or debugging container issues.
allowed-tools:
  - Bash
  - Read
  - Write
---

# Docker Container Management

## TRIGGER
Use this skill when:
- Writing or optimizing Dockerfiles
- Creating docker-compose.yml for multi-service apps
- Debugging container startup failures
- Managing volumes, networks, and environment variables

## Dockerfile Best Practices
```dockerfile
# Use specific version tags
FROM python:3.12-slim

# Set workdir early
WORKDIR /app

# Copy requirements first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source last
COPY . .

# Non-root user for security
RUN useradd -m appuser
USER appuser

EXPOSE 8000
CMD ["gunicorn", "myapp.wsgi:application", "--bind", "0.0.0.0:8000"]
```

## Docker Compose
```yaml
version: '3.9'
services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/mydb
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - .:/app

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes

volumes:
  postgres_data:
```

## Common Commands
```bash
# Build and run
docker build -t myapp:latest .
docker run -p 8000:8000 --env-file .env myapp:latest

# Compose
docker compose up -d          # Start detached
docker compose logs -f web    # Follow logs
docker compose exec web bash  # Shell into container
docker compose down -v        # Stop and remove volumes

# Debug
docker inspect <container_id>
docker stats                  # Resource usage
docker system prune -af       # Clean everything
```
"""
    },
    {
        "name": "nginx-reverse-proxy",
        "description": "Nginx configuration for reverse proxy, load balancing, and SSL termination. Use when configuring Nginx as a proxy for backend services.",
        "allowed_tools": ["Bash", "Read", "Write"],
        "content": """---
name: nginx-reverse-proxy
description: Nginx configuration for reverse proxy, load balancing, and SSL termination. Use when configuring Nginx as a proxy for backend services.
allowed-tools:
  - Bash
  - Read
  - Write
---

# Nginx Reverse Proxy Configuration

## TRIGGER
Use this skill when:
- Configuring Nginx to proxy requests to backend services
- Setting up blue/green deployment routing
- Adding SSL termination with Let's Encrypt
- Configuring rate limiting or caching headers

## Basic Reverse Proxy
```nginx
upstream backend {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;  # load balance
    keepalive 32;
}

server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_connect_timeout 10s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
}
```

## Blue/Green Routing
```nginx
# /etc/nginx/conf.d/active_color.conf
set $active_color "blue";

upstream blue  { server 127.0.0.1:8000; }
upstream green { server 127.0.0.1:8001; }

server {
    location / {
        proxy_pass http://$active_color;
    }
}
```

## Commands
```bash
nginx -t                        # Test config
nginx -s reload                 # Reload without downtime
tail -f /var/log/nginx/error.log   # Debug errors
systemctl status nginx          # Check service status
```
"""
    },
    {
        "name": "git-workflow",
        "description": "Git branching, commit, and code review workflows. Use when managing branches, resolving merge conflicts, or setting up Git hooks.",
        "allowed_tools": ["Bash"],
        "content": """---
name: git-workflow
description: Git branching, commit, and code review workflows. Use when managing branches, resolving merge conflicts, or setting up Git hooks.
allowed-tools:
  - Bash
---

# Git Workflow Management

## TRIGGER
Use this skill when:
- Creating feature branches or PRs
- Resolving merge conflicts
- Writing commit messages
- Squashing or rebasing commits before merge

## Branch Naming
```bash
git checkout -b feature/OCS-1234-add-risk-scoring
git checkout -b fix/OCS-5678-celery-timeout
git checkout -b hotfix/prod-null-pointer
```

## Clean Commit Messages
```
feat(risk): add velocity scoring for card transactions

- Calculate transaction frequency per 1h/24h/7d windows
- Flag accounts exceeding 10 transactions/hour
- Add configurable thresholds per tenant

Closes OCS-1234
```

## Common Operations
```bash
# Sync with main before PR
git fetch origin
git rebase origin/main

# Squash last 3 commits
git rebase -i HEAD~3

# Cherry-pick a fix to hotfix branch
git cherry-pick abc1234

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Stash and apply
git stash push -m "WIP: risk model changes"
git stash pop
```

## Conflict Resolution
```bash
git status                     # See conflicted files
git diff --diff-filter=U       # Show conflicts only
# Edit files to resolve, then:
git add resolved_file.py
git rebase --continue
```
"""
    },
    {
        "name": "pytest-testing",
        "description": "Python testing with pytest. Use when writing unit tests, integration tests, fixtures, or mocking external dependencies.",
        "allowed_tools": ["Bash", "Read", "Write"],
        "content": """---
name: pytest-testing
description: Python testing with pytest. Use when writing unit tests, integration tests, fixtures, or mocking external dependencies.
allowed-tools:
  - Bash
  - Read
  - Write
---

# Python Testing with Pytest

## TRIGGER
Use this skill when:
- Writing unit or integration tests
- Setting up test fixtures or factories
- Mocking external services (Redis, DB, APIs)
- Measuring test coverage

## Setup
```bash
pip install pytest pytest-django pytest-cov factory-boy
```

## Fixtures
```python
import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture
def transaction():
    return Transaction(
        id="txn_123",
        amount=5000.00,
        user_id=42,
        merchant_id="merch_001",
    )

@pytest.fixture
def mock_redis():
    with patch('myapp.cache.client') as mock:
        mock.get.return_value = None
        yield mock
```

## Writing Tests
```python
class TestRiskScoring:
    def test_high_amount_flags_transaction(self, transaction):
        transaction.amount = 100000.00
        score = calculate_risk_score(transaction)
        assert score > 0.8
        assert score <= 1.0

    def test_cache_miss_hits_database(self, mock_redis, transaction):
        mock_redis.get.return_value = None
        result = get_cached_score(transaction.id)
        mock_redis.get.assert_called_once_with(f"score:{transaction.id}")

    @pytest.mark.parametrize("amount,expected_flag", [
        (100, False),
        (50000, False),
        (200000, True),
    ])
    def test_amount_thresholds(self, amount, expected_flag, transaction):
        transaction.amount = amount
        assert is_high_value(transaction) == expected_flag
```

## Running Tests
```bash
pytest tests/ -v                          # Verbose
pytest tests/ --cov=myapp --cov-report=html  # Coverage
pytest tests/ -k "test_risk"              # Filter by name
pytest tests/ -x                          # Stop on first failure
pytest tests/ --tb=short                  # Short traceback
```
"""
    },
    {
        "name": "azure-devops-pipeline",
        "description": "Azure DevOps CI/CD pipeline configuration. Use when writing azure-pipelines.yml, setting up build/deploy stages, or debugging pipeline failures.",
        "allowed_tools": ["Bash", "Read", "Write"],
        "content": """---
name: azure-devops-pipeline
description: Azure DevOps CI/CD pipeline configuration. Use when writing azure-pipelines.yml, setting up build/deploy stages, or debugging pipeline failures.
allowed-tools:
  - Bash
  - Read
  - Write
---

# Azure DevOps Pipeline Configuration

## TRIGGER
Use this skill when:
- Writing or modifying azure-pipelines.yml
- Setting up multi-stage build and deploy pipelines
- Configuring SSH deployment tasks
- Debugging pipeline failures

## Pipeline Structure
```yaml
trigger:
  branches:
    include: [main, develop]

variables:
  IMAGE_TAG: $(Build.SourceVersion)
  ACR_REGISTRY: myregistry.azurecr.io

stages:
  - stage: Build
    jobs:
      - job: BuildAndPush
        pool:
          vmImage: ubuntu-latest
        steps:
          - task: Docker@2
            inputs:
              command: buildAndPush
              repository: ocs-backend
              tags: $(IMAGE_TAG)

  - stage: Deploy
    dependsOn: Build
    condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/main'))
    jobs:
      - deployment: DeployProd
        environment: production
        strategy:
          runOnce:
            deploy:
              steps:
                - task: SSH@0
                  inputs:
                    sshEndpoint: ssh-ocs-prod-general
                    runOptions: inline
                    inline: |
                      cd /opt/ocs
                      ./deploy_ocs.sh $(IMAGE_TAG)
```

## Debugging
```bash
# Check pipeline logs via CLI
az pipelines runs show --id <run-id>

# Re-run failed stage
az pipelines runs stage-list --run-id <id>
```
"""
    },
]


def load_raw_skills(path: str) -> list[dict]:
    """Load skills_raw.jsonl."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def to_mlx_pair(name: str, description: str, skill_content: str) -> dict:
    """Convert a skill into a single MLX training pair."""
    instruction = f'Generate a complete SKILL.md file for an AI agent skill called "{name}".'
    user_msg = f"{instruction}\n\nDescription: {description}"

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": skill_content.strip()},
        ]
    }


def main():
    print("=" * 55)
    print("  SkillForge Dataset Rebuilder")
    print("=" * 55)

    raw_path = Path("./skillvault_data/skills_raw.jsonl")
    if not raw_path.exists():
        print(f"❌ Not found: {raw_path}")
        return

    # ── Load raw skills ──
    all_skills = load_raw_skills(str(raw_path))
    print(f"\n📥 Loaded {len(all_skills)} raw skills")

    # ── Filter: trusted repos + min word count ──
    clean = []
    skipped_repo = 0
    skipped_words = 0

    for skill in all_skills:
        repo = skill.get("repo", "")
        words = skill.get("word_count", 0)
        name = skill.get("name", "")
        content = skill.get("raw", skill.get("content", ""))

        if repo not in TRUSTED_REPOS:
            skipped_repo += 1
            continue

        if words < MIN_WORDS:
            skipped_words += 1
            continue

        if not name or not content:
            continue

        clean.append(skill)

    print(f"\n🔍 Filtering results:")
    print(f"   Skipped (untrusted repo) : {skipped_repo}")
    print(f"   Skipped (< {MIN_WORDS} words)   : {skipped_words}")
    print(f"   Clean skills kept        : {len(clean)}")

    # ── Build training pairs from clean skills ──
    from_scraped = []
    for skill in clean:
        pair = to_mlx_pair(
            name=skill["name"],
            description=skill.get("description", skill["name"]),
            skill_content=skill.get("raw", skill.get("content", "")),
        )
        from_scraped.append(pair)

    print(f"\n📝 Training pairs from scraped skills: {len(from_scraped)}")

    # ── Add synthetic examples ──
    from_synthetic = []
    for s in SYNTHETIC_SKILLS:
        pair = to_mlx_pair(
            name=s["name"],
            description=s["description"],
            skill_content=s["content"],
        )
        from_synthetic.append(pair)

    print(f"✍️  Synthetic examples added          : {len(from_synthetic)}")

    # ── Combine ──
    all_pairs = from_scraped + from_synthetic
    random.seed(42)
    random.shuffle(all_pairs)

    # ── Train/val split (90/10) ──
    split = int(len(all_pairs) * 0.9)
    train_pairs = all_pairs[:split]
    val_pairs   = all_pairs[split:]

    # ── Save ──
    out_dir = Path("./skillvault_mlx")
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / "train.jsonl", "w") as f:
        for p in train_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    with open(out_dir / "valid.jsonl", "w") as f:
        for p in val_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 55}")
    print(f"  ✅  Dataset rebuilt")
    print(f"{'=' * 55}")
    print(f"  Train pairs : {len(train_pairs)}")
    print(f"  Val pairs   : {len(val_pairs)}")
    print(f"  Total       : {len(all_pairs)}")
    print(f"  Saved to    : {out_dir}/")

    print(f"""
📌 Now retrain from scratch:

python3 -m mlx_lm lora \\
  --model ./llama-3.2-3b \\
  --train \\
  --data $(pwd)/skillvault_mlx \\
  --iters 1000 \\
  --batch-size 2 \\
  --num-layers 16 \\
  --learning-rate 2e-4 \\
  --steps-per-report 10 \\
  --steps-per-eval 100 \\
  --val-batches 10 \\
  --save-every 100 \\
  --max-seq-length 2048 \\
  --grad-accumulation-steps 4 \\
  --adapter-path ./skillforge-adapters-v3
""")


if __name__ == "__main__":
    main()