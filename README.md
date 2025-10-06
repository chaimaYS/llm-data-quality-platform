# LLM-Powered Data Quality Platform

A generic data quality platform that profiles any dataset, computes the 8 canonical DQ dimensions, and uses LLMs to accelerate rule generation, semantic understanding, and anomaly explanation.

Works across **structured** (Postgres, Parquet, CSV, Excel), **semi-structured** (JSON), **PDFs** (native + scanned), and **images**.

## Architecture

```
                        ┌──────────────┐
                        │  FastAPI     │
                        │  REST API    │
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

## DQ Dimensions

All 8 dimensions scored uniformly across structured and unstructured data:

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

## Where the LLM fits

| Use case | Input | Output |
|----------|-------|--------|
| Semantic classifier | Column name + profile + samples | Semantic type (email, SSN, IBAN…), PII flag |
| Rule proposer | Schema + profile | Candidate rules as YAML with rationale |
| Anomaly explainer | Failing rows + rule | Plain-English root cause |
| Document classifier | Page image + text | Document class (invoice, contract, ID…) |
| Field extractor | Page images + schema | Structured fields with coordinates |
| Caption consistency | Image + caption | Match/mismatch with reasoning |

Rule of thumb: if a check can be SQL or regex, write SQL or regex. LLM is for discovery, mapping, and explanation.

## Project Structure

```
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
│   │   ├── client.py               # LLMClient abstraction + Claude/OpenAI adapters + caching
│   │   ├── semantic_classifier.py  # Column type classification + PII detection
│   │   ├── rule_proposer.py        # LLM-generated rule suggestions
│   │   └── anomaly_explainer.py    # Plain-English failure explanations
│   ├── multimodal/
│   │   ├── pdf_processor.py        # Native/scanned PDF: text + OCR + tables + page render
│   │   └── image_processor.py      # Quality metrics, blur, perceptual hash, EXIF
│   └── api/
│       └── server.py               # FastAPI endpoints
├── tests/
│   └── unit/
│       ├── test_profiling.py       # 13 tests
│       ├── test_rules.py           # 14 tests
│       └── test_scoring.py         # 12 tests
├── config/
│   ├── dimensions.yml              # Dimension weights
│   └── sample_rules.yml            # Example rules for a customer table
├── deploy/
│   └── kubernetes/
├── Dockerfile
├── Makefile
└── requirements.txt
```

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run tests
make test

# Profile a CSV file
python -m src.profiling.engine data/sample.csv

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

# Run profiling
curl -X POST http://localhost:8000/datasets/customers/profile

# Run full DQ check
curl -X POST http://localhost:8000/datasets/customers/run

# Get scores
curl http://localhost:8000/datasets/customers/scores
```

## Tech Stack

| Layer | Choice |
|-------|--------|
| Compute | DuckDB (small), Spark (large) |
| LLM gateway | LiteLLM (Claude, OpenAI, Bedrock) |
| Vision | Claude Vision, GPT-4o |
| PDF parsing | pdfplumber, PyMuPDF, Tesseract OCR |
| Image processing | Pillow, OpenCV, imagehash |
| API | FastAPI |
| Metadata | PostgreSQL + pgvector |
| PII redaction | Microsoft Presidio |
| Orchestration | Dagster |

## Author

**Chaima Yedes** — Principal Data & AI Architect
- [LinkedIn](https://www.linkedin.com/in/chaima-yedes/)
