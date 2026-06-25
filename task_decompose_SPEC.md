# Task Decomposition SPEC

## 1. Objective

Add a **task decomposition** step to the KRONOS pipeline that uses an LLM
(Groq / Llama-3.3-70B) to break complex questions into simpler sub-questions
**before** tree search runs.

**Why**: Complex questions (e.g. "Is there a cardiac finding in the left lung
that could explain the pleural effusion?") combine multiple reasoning steps.
The current parser classifies the question as one type, and tree search tries
to answer it in one pass. Decomposition splits this into atomic sub-questions
that the tree search engine already handles well (existential, relational,
negation, counting).

**Target user**: The KRONOS pipeline — transparent to the end user. Results
should be the same or better; never worse for simple questions (they skip
decomposition entirely).

## 2. Design

### Placement: Before tree search, after parsing

```
Question
  → Parser (classify type, extract target)
  → Decomposer (complex? → split into sub-questions)
      → for each sub-question: Parser → Tree Search → sub-answer
  → Aggregator (combine sub-answers into final answer)
  → SearchResult
```

Simple questions (high-confidence rule parse, single-intent) skip decomposition
entirely — zero extra tokens, zero latency.

### Complexity gate

A question is "complex" when ANY of:
- `parse_confidence < 1.0` (parser wasn't sure)
- Question contains multiple findings (2+ from finding_vocab)
- Question contains compound connectors: "and", "but", "because", "due to",
  "caused by", "associated with", "explain"
- Question text length > 120 characters (heuristic proxy)

The gate is rule-based (no LLM call). If the gate says "simple", skip
decomposition and run tree search directly as today.

### Decomposition prompt

Groq (Llama-3.3-70B) receives the original question and returns a JSON array
of sub-questions. Each sub-question is a plain English string that the existing
`QuestionParser` can handle.

```
Decompose this chest X-ray question into simpler sub-questions that can be
answered independently. Each sub-question should ask about ONE thing only.

Rules:
- Return a JSON array of strings: ["sub-q1", "sub-q2", ...]
- Each sub-question must be answerable from a single chest X-ray
- Keep the original clinical meaning — do not add or remove intent
- If the question is already simple, return it unchanged: ["<original>"]
- Maximum 4 sub-questions

Question: "{question}"
```

### Aggregation (LLM-based)

After tree search returns a `SearchResult` per sub-question, the aggregator
uses Groq to synthesize a final answer from the sub-answers + original question.

**Why LLM, not rules**: The connector between sub-questions matters — "X **and**
Y?" needs all-Yes, but "X **or** Y?" needs any-Yes. Causal questions ("does X
explain Y?") need cross-sub-question reasoning. A rule table can't handle this.

**Aggregation prompt**:
```
Given the original question and the sub-answers below, synthesize a single
final answer. Respect the logical connectors (and/or/because/due to).

Original question: "{original_question}"

Sub-answers:
{for each: "Q: {sub_q}  →  A: {sub_answer} (confidence: {tier})"}

Rules:
- Answer the original question directly and concisely
- If any sub-answer is uncertain or abstained, say so
- Do not add information beyond what the sub-answers provide
- Return JSON: {"answer": "<your answer>", "confidence": "high|medium|low"}
```

**Tier mapping**: The final tier is the *lowest* tier among sub-results
(conservative). If one sub-answer is ABSTAIN, the whole answer is ABSTAIN.
The LLM confidence field ("high"/"medium"/"low") is logged but does NOT
override the tier — the verifier remains the source of truth.

**Token cost**: 1 extra Groq call (~100 tokens in, ~50 out). Combined with
the decomposition call, a complex question costs 2 Groq calls total. Simple
questions still cost 0.

### Fallback

If Groq is unavailable (no API key, timeout, error), decomposition is skipped
and the question goes to tree search as-is. Same behavior as today.

## 3. Files to create / modify

### New file: `src/question/decomposer.py`

```python
# Core module. Three public functions:
#   is_complex(question, parse_confidence, finding_vocab) -> bool
#   decompose(question, llm_client) -> list[str]
#   aggregate(original_question, sub_questions, sub_results, llm_client) -> SearchResult
#
# decompose() and aggregate() each make 1 Groq call.
# is_complex() is pure rule-based (no LLM).
```

### Modified: `src/pipeline.py`

- Import decomposer
- In `run_with_facts()` and `run()`: after parsing, check `is_complex()`
- If complex: call `decompose()` → loop tree search → `aggregate()`
- If simple: tree search as today (no change)

### New file: `tests/test_decomposer.py`

- Test `is_complex()` with simple and complex examples
- Test `decompose()` with mocked LLM client
- Test `aggregate()` for each question type
- Test fallback when LLM returns garbage or is unavailable

## 4. Code style

- Follow CLAUDE.md: flat logic, plain functions, no abstractions for single use
- `is_complex()` is a plain function with if/elif checks
- `decompose()` calls `llm_client(prompt)`, parses JSON, returns list of strings
- `aggregate()` is a plain function with if/elif per question type
- No classes needed — three stateless functions

## 5. Testing strategy

| Test | Verifies |
|---|---|
| `test_simple_skips_decompose` | `is_complex("Is there Cardiomegaly?", 1.0, vocab)` returns False |
| `test_compound_is_complex` | "Is there Cardiomegaly and Pleural effusion?" → True |
| `test_low_confidence_is_complex` | parse_confidence=0.0 → True |
| `test_decompose_returns_subqs` | Mock LLM returns valid JSON → list of sub-questions |
| `test_decompose_fallback` | Mock LLM returns "" → returns [original_question] |
| `test_decompose_bad_json` | Mock LLM returns garbage → returns [original_question] |
| `test_aggregate_and_logic` | Mock LLM returns "Yes" for all-Yes sub-answers |
| `test_aggregate_or_logic` | Mock LLM returns "Yes" when any sub-answer is Yes (OR question) |
| `test_aggregate_fallback` | LLM unavailable → concatenate sub-answers as fallback |
| `test_aggregate_tier_conservative` | One ABSTAIN → final ABSTAIN regardless of LLM output |
| `test_pipeline_simple_unchanged` | Simple question → same result as before (no decomposition) |
| `test_pipeline_complex_decomposes` | Complex question → decompose called |

## 6. Boundaries

### Always do
- Skip decomposition for simple questions (zero overhead)
- Fall back gracefully when Groq is unavailable
- Preserve the exact same interface (`run()` / `run_with_facts()` return `SearchResult`)
- Log when decomposition happens (for debugging)

### Ask first
- Changing the Groq `max_tokens` (currently 128; decompose needs ~256, aggregate needs ~128)
- Adding new question types to `QType`
- Changing the complexity gate thresholds

### Never do
- Don't call Groq for simple questions
- Don't let decomposition change the answer for single-intent questions
- Don't add recursive decomposition (sub-questions decomposing further)
- Don't break the existing test suite
