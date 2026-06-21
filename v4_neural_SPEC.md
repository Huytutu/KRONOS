# SPEC — KRONOS v4 Neural Layer (LLaVA-Med Agent + Visual Tools)

> Scope: plug the **neural components** into the deterministic core built in
> `v4_core_SPEC.md`: LLaVA-Med 1.5 as the real agent (replacing MockAgent),
> and the 3 visual tools (`inspect`, `re_detect`, `compare`).
>
> **Prerequisite:** v4 deterministic core (contracts, verifier, symbolic tools,
> tree search) is built and passing (245 tests).
>
> References: [v4 intent](docs/intent/kronos-v4-multimodal-tree-search.md),
> [v4_core_SPEC.md](v4_core_SPEC.md).

---

## 1. Objective

Replace `MockAgent` with **LLaVA-Med 1.5** (`llava-med-v1.5-mistral-7b`) as
the real agent inside tree search, and add 3 **visual tools** that let the agent
look at the image during reasoning. The tree search, verifier, and symbolic
tools remain unchanged — only the agent and tool layer expand.

```
search(query, facts, dag, agent=LLaVAMedAgent, ...)  → SearchResult
                               ↑ was MockAgent, now real VLM
```

After this spec: the full v4 pipeline runs end-to-end on a CXR image + question
and returns answer + tier + trace with both visual and symbolic steps.

---

## 2. LLaVA-Med Agent (`src/agent/llavamed.py`)

### 2.1 Model

- **Model:** `llava-med-v1.5-mistral-7b` (LLaVA-Med 1.5, Mistral-7B base).
- **Loading:** use `transformers` + `llava` package. Support two precision modes
  via a `quantize` flag:
  - `quantize=False` → fp16 (needs ~14GB VRAM)
  - `quantize=True`  → 4-bit via `bitsandbytes` (needs ~6GB VRAM)
- **Frozen:** no fine-tuning, no LoRA, no training. Prompt-only.
- **Greedy decoding:** `temperature=0, do_sample=False` for reproducibility.

### 2.2 Interface

Implements the existing `Agent` Protocol:

```python
class LLaVAMedAgent:
    def __init__(self, model_path, image=None, quantize=False):
        # Load model + processor. Store image (PIL) for visual tools.

    def set_image(self, image):
        # Set/change the current image for visual actions.

    def propose_actions(self, node: TreeNode, query: Query, k: int) -> list[Action | str]:
        # Build prompt from (query, node.state_facts, node.history, node.reflection)
        # → ask LLaVA-Med to output k actions in structured format
        # → parse into Action objects or "Answer[...]" strings
```

### 2.3 Prompt design

The prompt is a **system + user** message. User message contains:
1. Question (natural language)
2. Current evidence graph (text: list of facts with concept/bbox/laterality/conf)
3. History so far (list of past action→observation pairs)
4. Available tools (name + signature + one-line description)
5. Reflection from failed sibling branch (if any)
6. Instruction: "Output up to {k} actions as JSON, or Answer[...] if ready."

**Output format** (VLM generates this, we parse):
```json
[
  {"tool": "is_a", "args": {"node": "cardiomegaly", "target": "cardiac_abnormality"}},
  {"tool": "inspect", "args": {"bbox": [100, 200, 300, 400]}}
]
```
or: `Answer[Yes]`

**Parsing:** regex/JSON extraction. If VLM output is malformed, return empty list
(search treats this as a dead branch and backtracks — graceful degradation).

### 2.4 Image handling

- The **original full image** is stored once at agent init (`set_image`).
- For `propose_actions`: the image is included in the VLM prompt as visual context
  (LLaVA-Med is a vision-language model). This lets the agent **see the image**
  when deciding which action to take.
- For visual tools (`inspect`, `compare`): the image is **cropped** to the
  specified bbox region before being passed to LLaVA-Med for detailed analysis.

---

## 3. Visual Tools (`src/tools/visual.py`)

Three tools. Each takes an `Action`, the original image, and returns an
`Observation`. Results are **folded into the evidence graph** as new/updated
`PerceptualFact` entries.

### 3.1 `inspect(bbox)` — zoom & describe

- **Who runs it:** LLaVA-Med on the cropped region.
- **Input:** `Action(tool="inspect", args={"bbox": [x1,y1,x2,y2]}, kind="visual")`
- **Process:** crop image to bbox → prompt LLaVA-Med: "Describe the abnormality
  in this region." → parse response into finding name + confidence.
- **Output:** `Observation(result={"concept": "consolidation", "conf": 0.75,
  "description": "..."}, ok=True)`.
- **Fold:** if `ok`, create a new `PerceptualFact` and add to `node.state_facts`.

### 3.2 `re_detect(region)` — run detector on sub-region

- **Who runs it:** YOLO detector (same as EvaX/perception, but on a crop).
- **Input:** `Action(tool="re_detect", args={"bbox": [x1,y1,x2,y2]}, kind="visual")`
- **Process:** crop image to bbox (with 20% padding) → resize to detector input
  size → run YOLO → map bbox back to original coordinates.
- **Output:** `Observation(result=[PerceptualFact, ...], ok=True)` — list of new
  findings in the region, or `ok=False` if nothing found.
- **Fold:** merge new facts into `node.state_facts` (no duplicates — deduplicate
  by IoU > 0.5 with existing facts).
- **This is the perception-repair action.** Agent calls it when it suspects EvaX
  missed something in a region.

### 3.3 `compare(r1, r2)` — compare two regions

- **Who runs it:** LLaVA-Med on two crops side-by-side.
- **Input:** `Action(tool="compare", args={"bbox1": [...], "bbox2": [...]}, kind="visual")`
- **Process:** crop both regions → prompt LLaVA-Med: "Compare these two regions.
  Which shows more opacity/abnormality?" → parse response.
- **Output:** `Observation(result={"comparison": "left region shows more opacity",
  "laterality_hint": "left"}, ok=True)`.
- **Fold:** does not create a fact directly; the result is used by the agent to
  inform the next symbolic action (e.g. `compose_laterality`).

### 3.4 Unified dispatch

Update `src/tools/symbolic.py` → rename to `src/tools/dispatch.py`:

```python
def run_tool(action, facts, dag, img_wh, image=None, detector=None, vlm=None):
    if action.kind == "visual":
        return run_visual_tool(action, image, detector, vlm)
    return run_symbolic_tool(action, facts, dag, img_wh)
```

Symbolic tools unchanged. Visual tools require `image` (PIL), `detector` (YOLO),
`vlm` (LLaVA-Med model/processor) — all passed from the search loop.

---

## 4. Integration: tree search with real agent

`search()` signature gains optional params:

```python
def search(query, facts, dag, agent, budget=20, k=3, img_wh=None,
           image=None, detector=None, vlm=None) -> SearchResult:
```

When visual tools are available (`image` is not None), `run_tool` can execute
both symbolic and visual actions. When `image` is None, visual actions return
`ok=False` (graceful fallback to symbolic-only — MockAgent still works).

**Fact folding:** after a visual tool returns new facts, the child node's
`state_facts` is updated (union, dedup by IoU). This expanded evidence is then
available to subsequent symbolic tools in the same branch.

---

## 5. End-to-end pipeline (`src/pipeline.py`)

Ties everything together:

```python
def run(image_path, question, dag, detector, agent, budget=20, k=3):
    image = load_image(image_path)
    facts = detector.detect(image_path)           # YOLO global scan (query-independent)
    query = parse_question(question)               # existing parser
    agent.set_image(image)
    result = search(query, facts, dag, agent, budget, k,
                    image=image, detector=detector.model, vlm=agent)
    return result
```

---

## 6. Testing strategy

### Unit tests (no GPU — mock VLM responses)

`tests/test_visual_tools.py`:
- `inspect` with mocked LLaVA-Med response → parses to PerceptualFact.
- `re_detect` with mocked YOLO → returns facts with correct bbox mapping.
- `compare` with mocked response → parses comparison.
- Malformed VLM output → `ok=False` (graceful).

`tests/test_llavamed_agent.py`:
- Prompt construction contains question + facts + tools + history.
- Mocked VLM output `[{"tool":"is_a",...}]` → parses to Action list.
- Mocked VLM output `Answer[Yes]` → returns string.
- Malformed output → returns empty list (dead branch).

`tests/test_pipeline.py`:
- With MockAgent + mocked detector → end-to-end returns SearchResult.

### Integration tests (GPU required — marked `@pytest.mark.gpu`)

- Full LLaVA-Med loaded, real image, real YOLO → end-to-end.
- These are skipped in CI without GPU. Run manually on server.

---

## 7. Code style

- Same as core: pydantic models, flat functions, no abstractions.
- VLM loading behind a factory function (`load_llavamed(path, quantize)`).
- All prompts as plain strings (no template engine).
- `@pytest.mark.gpu` on tests that need real model inference.
- Visual tools return `ok=False` when image/detector/vlm not available (never crash).

---

## 8. Boundaries

### Always
- LLaVA-Med is **frozen, prompt-only, greedy decoding**.
- Visual tool results **fold into graph** (become facts), verified by the
  same deterministic verifier.
- Visual actions **add/refine** evidence; never **replace** base EvaX evidence.
- Parse VLM output defensively — malformed → empty → dead branch → backtrack.

### Ask first
- Whether to add image-level caching (avoid re-cropping same bbox).
- Prompt engineering iterations (may need tuning for VLM to output valid JSON).
- Whether `re_detect` should use a confidence threshold different from global scan.

### Never
- Fine-tune or LoRA the VLM.
- Use VLM output as search reward (verifier-as-value stays deterministic).
- Let visual tools delete or override base EvaX facts.
- Run VLM during `verify()` — verification remains deterministic.
