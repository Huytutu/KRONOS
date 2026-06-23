# SPEC — Tiered Fallback for Parser and Linker (Groq API)

> Scope: add **LLM fallback tiers** to the two pipeline steps that currently
> only have rule-based implementations: `QuestionParser` and `ConceptLinker`.
> When the rule tier fails (returns `None` / low confidence), fall back to
> **Groq API** (fast cloud LLM) for a best-effort answer — but tag the
> output so the verifier knows the parse came from an LLM, not a rule.
>
> **Why Groq, not MedGemma?** MedGemma is frozen and dedicated to the tree
> search agent. Groq is fast (~100ms), free-tier, needs no GPU, and keeps
> the agent/fallback concerns cleanly separated.
>
> References: [v4_core_SPEC.md](v4_core_SPEC.md), existing code in
> `src/question/parser.py` and `src/linking/linker.py`.

---

## 1. Objective

The rule-based parser and linker are fast and deterministic, but brittle:

| Component | Current behavior | Failure mode |
|---|---|---|
| `QuestionParser` | Pattern-match on keywords (`"is there"`, `"how many"`, …) | Unusual phrasing → defaults to `relational` with `conf=0.0` |
| `ConceptLinker` | Exact synonym lookup in `synonyms.yaml` | Unseen spelling / abbreviation → returns `None` |

Both already carry a `tier` or `conf` field designed for this:
- `Query.parser_tier` is `"rule"` or `"llm"`
- `Query.parse_confidence` is `1.0` for rule hits, lower for fallback

**Goal:** when the rule tier fails, call Groq API (Llama-3 or similar fast
model) to recover a structured parse, tagged as `parser_tier="llm"`. The
verifier and search do not change — they already use `parse_confidence` and
can treat LLM-parsed queries more cautiously if needed.

**Groq setup:** API key in `.env` as `GROQ_API_KEY`. One thin client in
`src/llm/groq_client.py` wrapping the Groq REST API. No SDK dependency —
just `requests` (already available) or `httpx`.

---

## 2. Design: two-tier cascade

```
Input text
   │
   ▼
[Tier 1] Rule-based (current code, fast, deterministic)
   │
   ├── success (conf ≥ 1.0) ─────────► return result
   │
   ▼
[Tier 2] LLM fallback (MedGemma, slower, best-effort)
   │
   ├── parseable response ──► return result with parser_tier="llm", conf=0.5
   │
   ▼
   return original tier-1 result (even if low-conf)
```

### 2a. QuestionParser fallback

**When to trigger:** the rule tier sets `parse_confidence=0.0` (the `else`
branch at line 42 — unknown wording).

**LLM prompt** (few-shot, text-only, no image needed):

```
Classify this chest X-ray question.
Return exactly one JSON object: {"type": "<type>", "target": "<finding or null>"}

Types: existential, negation, relational, counting, open

Examples:
Q: "Is there Cardiomegaly?"         → {"type": "existential", "target": "Cardiomegaly"}
Q: "Are the lungs clear?"           → {"type": "negation", "target": null}
Q: "Where is the Pleural effusion?" → {"type": "relational", "target": "Pleural effusion"}
Q: "How many findings are there?"   → {"type": "counting", "target": null}
Q: "What abnormality is visible?"   → {"type": "open", "target": null}

Q: "{question}"
```

**Parsing the response:** extract JSON with regex `\{.*\}`, validate `type`
is one of the 5 known types, validate `target` against finding vocab (or
pass to linker). If anything fails → keep the tier-1 result unchanged.

**Output:** `Query(type=..., target=..., parse_confidence=0.5, parser_tier="llm")`

### 2b. ConceptLinker fallback

**When to trigger:** `link()` returns `None` (no exact synonym match).

**LLM prompt** (few-shot, text-only):

```
Map this medical finding to one of the canonical names below.
Return exactly the canonical name, nothing else.

Canonical names: Aortic enlargement, Atelectasis, Calcification, Cardiomegaly,
Consolidation, ILD, Infiltration, Lung Opacity, Nodule/Mass, Other lesion,
Pleural effusion, Pleural thickening, Pneumothorax, Pulmonary fibrosis

Finding: "{text}"
```

**Parsing the response:** strip whitespace, check if the response is one of
the 14 canonical names (case-insensitive). If not → return `None` (same as
before, no hallucinated mapping).

**Output:** the canonical name string, or `None`.

---

## 3. Implementation plan

### Step 1 — `QuestionParser`: add LLM fallback

File: `src/question/parser.py`

- Add `_llm_parse(self, question) -> Query | None` method.
- In `parse()`, after the `else` branch (conf=0.0), if `self.llm_client`
  is not None, call `_llm_parse`. If it returns a valid Query, use it.
- `_llm_parse` builds the few-shot prompt, calls
  `self.llm_client(prompt)`, parses JSON from the response, validates
  type and target, returns a Query with `parser_tier="llm"`, `conf=0.5`.
- If parsing fails at any point, return `None` (keeps tier-1 result).
- `llm_client` is a callable `(str) -> str`. In production this is
  `groq_client.complete`; in tests it's a mock lambda.
  Injected via constructor (already has the `llm_client` param).

**Verify:** `pytest tests/test_parser.py tests/test_parser_gold.py` — all
existing tests pass unchanged (they don't inject an llm_client).

### Step 2 — `ConceptLinker`: add LLM fallback

File: `src/linking/linker.py`

- Add `llm_client=None` to `__init__` and store it.
- Add `_llm_link(self, text) -> str | None` method.
- In `link()`, if exact lookup returns `None` and `self.llm_client` is
  not None, call `_llm_link`. Validate the response against the 14
  canonical names. Return canonical name or `None`.
- `llm_client` is the same `(str) -> str` callable.

**Verify:** `pytest tests/test_linker.py tests/test_linker_gold.py`.

### Step 3 — Groq client

File: `src/llm/groq_client.py` (new)

- Thin wrapper: `complete(prompt, model="llama-3.3-70b-versatile") -> str`.
- Reads `GROQ_API_KEY` from env (via `python-dotenv`, already in deps).
- Uses `requests.post` to `https://api.groq.com/openai/v1/chat/completions`.
- `temperature=0`, `max_tokens=128` (we only need a short JSON response).
- On any error (network, rate limit, bad response) → return `""`.

```python
def complete(prompt, model="llama-3.3-70b-versatile"):
    ...
```

**Verify:** quick smoke test calling the API with a simple prompt.

### Step 4 — Wire into pipeline

File: `src/pipeline.py`

- At module level (lazy): `from src.llm.groq_client import complete as groq_complete`.
- The module-level `_parser` already exists. Add a `_get_parser()` that
  creates a parser with `llm_client=groq_complete` if the API key is set,
  else `llm_client=None`.
- The `run_with_facts()` path keeps `llm_client=None` → pure rule-based.

**Verify:** `pytest tests/test_pipeline.py` + manual notebook test.

### Step 5 — Add tests for fallback paths

File: `tests/test_parser.py` (extend), `tests/test_linker.py` (extend)

- Test: parser with mocked `llm_client` returning valid JSON → produces
  Query with `parser_tier="llm"`, correct type/target.
- Test: parser with mocked `llm_client` returning garbage → falls back
  to rule tier-1 result.
- Test: linker with mocked `llm_client` returning canonical name → works.
- Test: linker with mocked `llm_client` returning unknown name → returns
  `None`.

---

## 4. Boundaries

**Always:**
- Tag LLM-parsed output with `parser_tier="llm"` and `conf ≤ 0.5`.
- Validate LLM responses against known types/names — never trust raw.
- Keep existing rule-based behavior unchanged for all current test cases.

**Ask first:**
- Whether to lower the verifier's trust for LLM-parsed queries (e.g.
  cap at tier B instead of tier A). Currently not planned — the verifier
  already checks the DAG regardless of how the query was parsed.

**Never:**
- Let the LLM invent a question type or finding name not in the vocab.
- Call the LLM when the rule tier already matched (conf ≥ 1.0).
- Change the verifier or tree search logic — fallback is only for parsing.

---

## 5. Testing strategy

| Test | What it proves |
|---|---|
| Existing parser/linker tests (no llm_client) | Rule tier unchanged |
| Mock llm_client → valid JSON | Fallback produces correct Query |
| Mock llm_client → garbage | Graceful degradation to tier-1 |
| Mock llm_client → hallucinated type | Rejected, tier-1 kept |
| Mock llm_client → hallucinated finding | Rejected, returns None |
| Integration: `run()` with agent on unknown question phrasing | End-to-end fallback works |

---

## 6. File changes summary

| File | Change |
|---|---|
| `src/llm/__init__.py` | Empty init |
| `src/llm/groq_client.py` | New: thin Groq API wrapper (`complete(prompt) -> str`) |
| `src/question/parser.py` | Add `_llm_parse`, call it on conf=0.0 |
| `src/linking/linker.py` | Add `llm_client` param, `_llm_link` method |
| `src/pipeline.py` | Wire `groq_client.complete` as llm_client |
| `tests/test_parser.py` | Add fallback tests with mocked llm_client |
| `tests/test_linker.py` | Add fallback tests with mocked llm_client |
