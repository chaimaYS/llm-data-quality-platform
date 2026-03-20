# LLM-Powered Data Quality Platform

A generic data quality platform that profiles any dataset, computes the 8 canonical DQ dimensions, and uses LLMs to accelerate rule generation, semantic understanding, and anomaly explanation.

Works across **structured** (Postgres, Parquet, CSV, Excel), **semi-structured** (JSON), **PDFs** (native + scanned), and **images**.

## Architecture

```
                        ┌──────────────┐
                        │  Streamlit   │
                        │  UI / API    │
                        └──────┬───────┘
                               │
                  ┌────────────▼────────────┐
                  │     Orchestrator        │
                  │  (register → profile    │
                  │   → rules → score)      │
                  └────────────┬────────────┘
                               │
     ┌─────────────┬───────────┼───────────┬──────────────┐
     │             │           │           │              │
┌────▼────┐  ┌─────▼─────┐ ┌──▼──────┐ ┌──▼─────┐  ┌─────▼─────┐
│Connector│  │ Profiling  │ │Multimod.│ │ Rules  │  │   LLM     │
│  Layer  │  │  Engine    │ │Processor│ │ Engine │  │  Service   │
│(PG,File)│  │ (DuckDB)  │ │(PDF/Img)│ │(YAML)  │  │(text+vision)│
└─────────┘  └─────┬─────┘ └────┬────┘ └───┬────┘  └───────────┘
                   │            │          │
                   ▼            ▼          ▼
            ┌──────────────────────────────────┐
            │  Results Store (Postgres)        │
            │  Profiles, scores, history       │
            └──────────────────────────────────┘
```

## How Scoring Works

### The pipeline

Every dataset goes through the same pipeline, regardless of whether it's a database table, a CSV, a PDF, or an image collection:

```
Raw data  →  Profile (stats per column)
                  ↓
            Rules evaluate (pass/fail per rule)
                  ↓
            Dimension scores (weighted pass rate)
                  ↓
            Overall score + letter grade
                  ↓
            Dashboard / alerts / SLA gate
```

### The 8 quality dimensions

These are the industry-standard dimensions from DAMA and ISO 8000. Every dataset is scored against all 8:

| Dimension | What it measures | Example check |
|-----------|-----------------|---------------|
| **Completeness** | Are required values present? | `email` must not be null |
| **Uniqueness** | No unintended duplicates? | `customer_id` must be unique |
| **Validity** | Values match expected format? | `age` must be 0–120; `email` must match regex |
| **Consistency** | Values agree across fields? | `city` must match `country` |
| **Timeliness** | Data is fresh enough? | `last_updated` within 24 hours |
| **Accuracy** | Matches source of truth? | `balance` matches ERP system |
| **Integrity** | Referential integrity holds? | `product_id` must exist in products table |
| **Conformity** | Follows standards? | `phone` must be E.164 format |

### How a dimension score is calculated

Each dimension has one or more rules. Each rule evaluates every row and produces a pass count and fail count. The dimension score is the **weighted pass rate** of its rules, where severity determines the weight:

```
Example: Completeness dimension

Rule 1 — email null check:    475/500 passed = 95.0%  (HIGH severity → weight 3)
Rule 2 — country null check:  450/500 passed = 90.0%  (MEDIUM severity → weight 2)  
Rule 3 — status null check:   460/500 passed = 92.0%  (LOW severity → weight 1)

Completeness score = (95% × 3  +  90% × 2  +  92% × 1) / (3 + 2 + 1) = 92.7%
```

HIGH-severity failures weigh 3×. This means a critical field being null matters more than an optional field.

### How the overall score is calculated

The overall dataset score is the **weighted average of all 8 dimension scores**. Default weights are configurable in `config/dimensions.yml`:

```yaml
# config/dimensions.yml
completeness:  0.20    # Most impactful — missing data breaks everything
uniqueness:    0.15
validity:      0.15
consistency:   0.10
timeliness:    0.10
accuracy:      0.10
integrity:     0.10
conformity:    0.10
```

Example calculation:

```
Completeness:  92.7% × 0.20 = 18.5%
Uniqueness:   100.0% × 0.15 = 15.0%
Validity:      96.0% × 0.15 = 14.4%
Consistency:  100.0% × 0.10 = 10.0%  ← no rules → assumed 100%
Timeliness:   100.0% × 0.10 = 10.0%
Accuracy:     100.0% × 0.10 = 10.0%
Integrity:    100.0% × 0.10 = 10.0%
Conformity:   100.0% × 0.10 = 10.0%
─────────────────────────────────────
Overall:                       97.9% → Grade A
```

Dimensions with no rules get 100% (no evidence of problems). As you add rules, the score reflects reality.

### Letter grades

| Grade | Score | Meaning | Action |
|-------|-------|---------|--------|
| **A** | ≥ 95% | Production-ready | No action needed |
| **B** | ≥ 85% | Usable with minor issues | Address when convenient |
| **C** | ≥ 70% | Needs attention | Schedule fixes this sprint |
| **D** | ≥ 50% | Significant problems | Escalate to data owner |
| **F** | < 50% | Unreliable | Do not use for decisions |

### Why configurable weights matter for governance

Different teams have different priorities:

| Team | They care most about | Weight adjustment |
|------|---------------------|-------------------|
| Finance | Accuracy, Consistency | Accuracy → 30%, Consistency → 20% |
| Marketing | Timeliness, Completeness | Timeliness → 25%, Completeness → 25% |
| Compliance | Conformity, Validity | Conformity → 25%, Validity → 20% |
| Operations | Timeliness, Integrity | Timeliness → 25%, Integrity → 20% |

Each domain configures its own weights. Same scoring engine, different priorities.

### Drift detection

Every run is stored. Compare scores over time:

```
Run 1 (Jan 5):  Completeness 98%  →  Grade A
Run 2 (Jan 12): Completeness 95%  →  Grade A
Run 3 (Jan 19): Completeness 85%  →  Grade B  ← something changed upstream
Run 4 (Jan 26): Completeness 72%  →  Grade C  ← alert: degradation trend
```

The platform detects score drops and alerts before dashboards break.

### SLA enforcement

Governance policies become automated gates:

- "No dataset below Grade B goes to the production warehouse"
- "Alert the data owner if any dimension drops below 80%"
- "Block dashboard refresh if Timeliness < 90%"

## DQ Dimensions — Structured vs Unstructured

The same 8 dimensions apply to every data type. The checks are different, but the scores are unified:

| Dimension | Structured | PDFs | Images |
|-----------|-----------|------|--------|
| **Completeness** | Null counts, required fields | Expected pages/fields present | Required metadata present |
| **Uniqueness** | Distinct count, PK check | Text hash dedup | Perceptual hash near-dedup |
| **Validity** | Regex, type cast, enum | OCR confidence, file opens | Resolution, blur, MIME check |
| **Consistency** | Cross-field rules | Internal cross-refs | Caption matches content |
| **Timeliness** | max(updated_at) vs SLA | Document date vs SLA | EXIF DateTime vs expected |
| **Accuracy** | Reference reconciliation | Extracted fields vs ground truth | Declared vs detected attributes |
| **Integrity** | FK checks | Linked attachments resolve | Manifest matches file set |
| **Conformity** | ISO format validation | Template match | Channel standards |

A downstream consumer sees one API, one score, one grade — regardless of whether the source was a Postgres table or a folder of scanned invoices.

## Where the LLM Fits

Rule of thumb: **if a check can be SQL or regex, write SQL or regex**. LLM is for discovery, mapping, and explanation — the tasks where deterministic code falls short.

| Use case | Input | Output | When to use |
|----------|-------|--------|-------------|
| Semantic classifier | Column name + profile + samples | Semantic type (email, SSN, IBAN…), PII flag | Onboarding a new dataset |
| Rule proposer | Schema + profile | Candidate rules as YAML with rationale | Bootstrapping rules for a new table |
| Anomaly explainer | Failing rows + rule | Plain-English root cause | Alert messages, incident reports |
| Document classifier | Page image + text | Document class (invoice, contract, ID…) | PDF ingestion |
| Field extractor | Page images + schema | Structured fields with coordinates | Document accuracy checks |
| Caption consistency | Image + caption | Match/mismatch with reasoning | Media library quality |

Every LLM call is **cached** (hash of prompt + model → result, 30-day TTL) to minimize cost. Vision calls are **sampled** (not every document, every run).

## Project Structure

```
├── app.py                          # Streamlit UI — upload, profile, score
├── src/
│   ├── connectors/
│   │   ├── base.py                 # Abstract Connector interface
│   │   ├── postgres.py             # PostgreSQL via SQLAlchemy
│   │   └── file_connector.py       # CSV/Parquet/Excel via DuckDB
│   ├── profiling/
│   │   └── engine.py               # Per-column stats via DuckDB
│   ├── rules/
│   │   ├── base.py                 # Rule/RuleResult/Dimension abstractions
│   │   ├── builtin.py              # NullCheck, UniqueCheck, RangeCheck, RegexCheck, FK, Freshness
│   │   └── engine.py               # Load rules from YAML, evaluate, collect results
│   ├── scoring/
│   │   └── scorer.py               # Dimension scoring with configurable weights
│   ├── llm/
│   │   ├── client.py               # LLMClient abstraction + caching + cost tracking
│   │   ├── semantic_classifier.py  # Column type classification + PII detection
│   │   ├── rule_proposer.py        # LLM-generated rule suggestions
│   │   └── anomaly_explainer.py    # Plain-English failure explanations
│   ├── multimodal/
│   │   ├── pdf_processor.py        # Native/scanned PDF: text + OCR + tables + page render
│   │   └── image_processor.py      # Quality metrics, blur, perceptual hash, EXIF
│   └── api/
│       └── server.py               # FastAPI REST endpoints
├── tests/                          # 89 unit tests
│   └── unit/
│       ├── test_profiling.py       # Profiling engine (13 tests)
│       ├── test_rules.py           # Built-in rules (14 tests)
│       ├── test_scoring.py         # Dimension scoring (12 tests)
│       ├── test_connectors.py      # File connector (6 tests)
│       ├── test_multimodal.py      # Image processor (16 tests)
│       ├── test_llm.py            # LLM client + classifiers (10 tests, mocked)
│       ├── test_pipeline.py        # End-to-end pipeline (9 tests)
│       └── test_api.py            # FastAPI endpoints (3 tests)
├── config/
│   ├── dimensions.yml              # Dimension weights (customizable per team)
│   └── sample_rules.yml            # Example rules for a customer table
├── data/
│   └── sample_customers.csv        # Sample dataset for testing
├── Dockerfile
├── Makefile
└── requirements.txt
```

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run tests (89 tests)
make test

# Launch the UI
streamlit run app.py
# → http://localhost:8501
# Upload data/sample_customers.csv to see a quality report

# Start the API
make serve
# → http://localhost:8000/docs
```

### API Usage

```bash
# Register a dataset
curl -X POST http://localhost:8000/datasets \
  -H "Content-Type: application/json" \
  -d '{"name": "customers", "source_type": "file", "connection": {"path": "data/customers.csv"}}'

# Run full DQ check
curl -X POST http://localhost:8000/datasets/customers/run

# Get scores
curl http://localhost:8000/datasets/customers/scores
```

## Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Compute | DuckDB (small), Spark (large) | DuckDB is fast and embeddable; Spark for scale |
| LLM gateway | LiteLLM | Provider abstraction (Claude, OpenAI, Bedrock) |
| Vision | Claude Vision, GPT-4o | Document understanding, caption consistency |
| PDF parsing | pdfplumber, PyMuPDF, Tesseract | Native-text + layout + OCR fallback |
| Image processing | Pillow, OpenCV | Quality metrics, blur detection, perceptual hash |
| API | FastAPI | Typed endpoints, auto-generated OpenAPI docs |
| UI | Streamlit | Fast to build, file upload, interactive charts |
| Metadata | PostgreSQL + pgvector | Relational storage + vector search for embeddings |
| PII redaction | Microsoft Presidio | Pre-LLM safety layer |
| Orchestration | Dagster | Asset-based scheduling, fits the DQ mental model |

## Author

**Chaima Yedes** — Principal Data & AI Architect
- [LinkedIn](https://www.linkedin.com/in/chaima-yedes/)
