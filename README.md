# Codebase Knowledge Analyzer

A token-aware, LLM-assisted static-analysis pipeline built for the
[`spring-rest-sakila`](https://github.com/codejsha/spring-rest-sakila) coding
assignment. The analyzer is intentionally separate from the target repository:
the Sakila project remains an unmodified input.

## Assignment goals

The pipeline:

1. Read a repository efficiently and exclude generated, binary, oversized, and
   irrelevant files.
2. Parse source at class and method boundaries rather than using arbitrary
   character chunks.
3. Calculate deterministic facts such as signatures, source locations, lines
   of code, and cyclomatic complexity.
4. Use an LLM only for semantic interpretation: descriptions, architecture,
   workflows, and noteworthy design decisions.
5. Respect a configurable token budget for every request.
6. Aggregates method and file results hierarchically into a repository overview.
7. Validate the final artifact against a strict, versioned Pydantic model and
   emit readable JSON.

This hybrid design avoids asking an LLM to calculate facts that static tooling
can produce more accurately. It also avoids relying solely on retrieval, which
could omit files and methods from a repository-wide deliverable.

## Architecture

```text
Repository
    |
    v
Deterministic scanner ----> content hashes / cache keys
    |
    v
Java-aware parser --------> classes, methods, annotations, locations
    |
    +----------------------> complexity metrics
    |
    v
Token-aware chunk planner
    |
    v
Deterministic model router
    |-- routine chunks ------> GPT-5.6 Luna
    |-- complex chunks ------> GPT-5.6 Terra
    |
    v
Method -> file -> module -> project aggregation --> GPT-5.6 Sol synthesis
    |
    v
Pydantic validation ------> output/sakila-analysis.json
```

The primary analysis covers committed source. Build-generated QueryDSL and
MapStruct classes are excluded by default because they are implementation
artifacts rather than authored source.

## Implementation status

The implemented application includes:

- A deterministic scanner with stable ordering and SHA-256 file hashes.
- Explicit source/context classification and skip accounting.
- Generated-directory, symlink, file-size, and optional test exclusions.
- Java declaration extraction with `javalang` and a documented fallback for
  newer syntax.
- Cyclomatic complexity and logical-line measurements with Lizard.
- Model-aware token counting and method-boundary chunking with bounded overlap.
- Auditable model routing: Luna for routine extraction, Terra for complex
  extraction, and Sol for module/repository synthesis.
- LangChain `ChatOpenAI` integration with native structured output.
- Strict intermediate and final Pydantic schemas.
- Bounded concurrency, exponential retry, prompt-injection instructions, and
  content-addressed response caching.
- Hierarchical file, module, and repository synthesis.
- Atomic JSON writes to avoid incomplete output.
- Credential-free manifest, inventory, and JSON Schema commands.
- Unit, integration, strict typing, lint, and coverage checks.

## Requirements

- Python 3.11 or newer
- An OpenAI API key for the `analyze` command
- The target repository checked out locally

Java, MySQL, Redis, and Docker are not needed for static analysis. Java 17 may
later be used to compile the target as an optional validation step.

## Local setup

From the `codebase-analyzer` directory:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
cp .env.example .env
```

Add `OPENAI_API_KEY` to `.env` before running `analyze`. The model configuration
uses:

```text
CODE_ANALYZER_ROUTINE_MODEL=gpt-5.6-luna
CODE_ANALYZER_COMPLEX_MODEL=gpt-5.6-terra
CODE_ANALYZER_SYNTHESIS_MODEL=gpt-5.6-sol
```

The `.env` file is ignored by Git.

## Commands

Create a deterministic Sakila repository manifest:

```bash
codebase-analyzer scan ../spring-rest-sakila \
  --output output/sakila-manifest.json
```

Measure parser coverage, complexity, token demand, and planned model routing
without credentials:

```bash
codebase-analyzer inspect ../spring-rest-sakila \
  --output output/sakila-inventory.json
```

Generate the complete LLM-assisted report:

```bash
codebase-analyzer analyze ../spring-rest-sakila \
  --output output/sakila-analysis.json
```

The first analysis populates `output/cache`. Repeating the command with
unchanged files, selected model, and prompt version reuses validated responses.

Export the final output contract as JSON Schema:

```bash
codebase-analyzer schema --output output/analysis-schema.json
```

Run quality checks:

```bash
ruff check .
mypy src
pytest --cov
```

## Token-limit strategy

The pipeline reserves part of the model context for output and prompt
instructions:

```text
available prompt tokens = model context limit - reserved output tokens
```

Each code chunk must fit inside a smaller `chunk_token_budget`, leaving room for
the extraction prompt and surrounding metadata. Oversized types are split at
method boundaries. Oversized methods use a documented fallback split with
source-line overlap. No request is sent until its token count is checked.

Repository-wide summaries use hierarchical reduction instead of placing the
entire repository into one prompt. Aggregate and per-model token usage are
recorded in the final metadata.

## Output contract

The final JSON report is designed for both people and programs. It includes:

- Reproducibility metadata, repository revision, all three models, routing
  counts/reasons, file counts, and aggregate/per-model token usage.
- A high-level purpose, architecture, technology, workflow, complexity, and
  limitation overview.
- Module descriptions and dependencies.
- File summaries with content hashes.
- Types and methods with signatures, source locations, annotations, complexity
  metrics, semantic descriptions, and noteworthy aspects.

Unknown information is represented explicitly instead of being invented.
Unexpected JSON fields are rejected during validation.

## Key design decisions

### Python instead of Spring Boot

The assignment does not prescribe a language. Python was selected because
LangChain, Pydantic, tokenizers, and code-analysis libraries integrate directly.
The analyzer is a batch CLI, so a Spring web container would add operational
weight without improving the analysis. Keeping it separate also demonstrates
that the target repository is immutable input.

### Static facts before LLM interpretation

The scanner and parser own file identity, signatures, annotations, source
locations, and complexity. The LLM receives those facts as a catalog and adds
semantic descriptions. The final merger only accepts descriptions for known
catalog entries, reducing hallucinated methods.

### Complete traversal instead of retrieval-only RAG

Retrieval is useful for questions but can omit unselected methods. This
deliverable needs repository-wide coverage, so every eligible file is processed
once and then reduced hierarchically.

### Tiered model routing

One model is not equally cost-effective for every stage. A deterministic router
uses source facts available before any LLM request:

- Luna handles routine DTOs, entities, repositories, tests, and simple methods.
- Terra handles custom query implementations, security/configuration code,
  parser-warning files, large chunks, SQL/build context, and complex business
  control flow. Generated-style `equals`, `hashCode`, and `toString` complexity
  does not trigger escalation by itself.
- Sol synthesizes validated file evidence into modules and the repository
  overview.

The chosen model is included in every cache key. Routing counts and escalation
reasons are written to report metadata, keeping cost and quality decisions
auditable.

### Preflight before paid requests

`inspect` proves that the repository parses and reports the exact number and
size of planned chunks, including the Luna/Terra split and every escalation
reason. For the Sakila revision used here it selected 213 files, identified 265
types and 652 methods, and planned 213 chunks totaling 102,843 source tokens:
170 routine Luna chunks, 43 complex Terra chunks, and 14 Sol synthesis
artifacts.

## Assumptions and limitations

- Static analysis cannot prove runtime behavior involving Spring dependency
  injection, proxies, Redis caching, JWT filters, or database queries.
- Lombok, MapStruct, and QueryDSL generate behavior not visible in committed
  source. The report will identify this limitation instead of pretending those
  implementations were analyzed.
- `javalang` targets Java 8 grammar. A Lizard-backed fallback preserves method
  extraction for files containing newer syntax and marks those files with
  `has_parser_warnings`.
- Cyclomatic complexity is calculated by deterministic tooling; semantic
  complexity assessments from the LLM are clearly separated.
- LLM output may vary between model versions. Strict schemas, explicit reasoning
  effort, caching, and recorded model/routing metadata reduce that variability.
