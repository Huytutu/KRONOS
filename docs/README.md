# KRONOS — Design Document

**Knowledge-Routed Neuro-Symbolic Multimodal Reasoning for Faithful Chest X-ray VQA**

> Status: design v2 (propose-then-verify). Supersedes the E-SyM planner-loop draft.
> Target venue: A*-rank (CVPR / ICCV / MICCAI / NeurIPS D&B or main).

---

## 1. Thesis

Medical VQA systems are optimized for **answer accuracy** while ignoring **evidence faithfulness** —
whether the answer was reached *for the right, clinically relevant reasons*. A model can be right by
shortcut. This project's single selling point:

> Answer correctly **and** emit a trace that is the *actual cause* of the answer, not a post-hoc
> rationalization — by deriving the answer with a symbolic engine whose derivation **is** the trace,
> using neural models only at the edges.

Everything in this design is subordinate to keeping that claim defensible under adversarial review.

---

## 2. Design principles (non-negotiable)

1. **Neural at the edges, symbolic at the deciding core.** The LLM parses the question, links concepts,
   and (optionally) proposes *where to look*. It never produces the conclusion or the trace.
   - *Parsing, not decomposition.* The LLM **translates** one question into a fixed closed schema
     `⟨type, target, constraints⟩` — it does not break the question into reasoning sub-steps. The
     schema has no slot for an answer, so the parse cannot smuggle the conclusion in. A clinical rule
     (e.g. "CTR > 0.5 ⇒ cardiomegaly") must live in the engine, never in the LLM's parse.
   - *Perception is query-independent — on purpose.* Findings are facts about the **image**, extracted
     before the query is consulted. Query-conditioned captioning would let the question prime what the
     model "sees" (the exact shortcut we are defending against) and would break closed-world negation,
     which needs *all* findings, not just query-relevant ones.
2. **The trace is engine-derived, never LLM-narrated.** Faithfulness is *by construction* (the answer
   provably follows from cited facts), not a claim about a self-reported chain of thought.
   - *Faithfulness ≠ correctness (scope honesty).* KRONOS guarantees the answer follows from the cited
     facts. It does **not** guarantee the facts match the image — that is bounded by the frozen vision
     model. A wrong perception yields a *faithful-but-wrong* answer. The design makes such errors
     **visible, attributable, and partially hedged** (see §6), not hidden behind a fabricated rationale.
3. **Ontology is a reasoning layer over perceptual facts, not an answer store.**
   `answer = query over (perceptual facts ∪ ontology axioms)`. The KG never "knows" the patient.
4. **Selection is grounded in verification, not learned guessing.** Heads *propose*; the engine
   *verifies*; the surviving derivation is the answer. (This replaces the v1 top-1 router-classifier.)
5. **Principled abstention over confident guessing.** When nothing verifies and confidence is low,
   abstain — and report selective-risk, not raw top-1 accuracy.
6. **Frozen backbones, reproducible, cheap.** The only trained component is a small router used for
   *efficiency/tie-break*. The engine is CPU-only inference over a precomputed DAG.

---

## 3. Contributions

| # | Contribution |
|---|---|
| C1 | **Propose-then-verify reasoning core.** Heterogeneous heads propose candidate answers; an *expert-agnostic symbolic verifier* checks each against `(facts ∪ ontology axioms)`; the emitted answer carries the engine's verification derivation as its trace. A small faithfulness-supervised router handles ordering and tie-break only. |
| C2 | **Question-typed symbolic engine over a curated ontology DAG.** Termination is owned by the engine via *closure conditions* derived from the question type; negation/absence questions are answered under **closed-world enumeration** on a static, hand-curated exclusion list. Coverage gaps trigger principled abstention. |
| C3 | **Causal faithfulness evaluation.** EF@k scores the *derivation subgraph* (not the search trajectory) against gold evidence regions, **plus a deletion/perturbation test** showing the answer flips when a fact/edge in the trace is removed — evidence the trace is causal, not decorative. |

---

## 4. Architecture

```text
                          Medical Image + Question
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        ▼                                                         ▼
 ┌──────────────────┐                                   ┌──────────────────┐
 │ E_perc           │  PERCEPTION FRONT-END (shared)    │ Question Parser  │  (LLM, edge)
 │ Vision / Med-MLLM│                                   │ q → structured   │
 └────────┬─────────┘                                   │   query + type   │
          │ grounded findings                           └────────┬─────────┘
          │ (concept, region, laterality, conf)                  │
          ▼                                                      │
 ┌──────────────────┐                                            │
 │ Concept Linking  │  (LLM/dict, edge)                          │
 │ finding → node   │  → report linking_accuracy separately      │
 └────────┬─────────┘                                            │
          │ canonical evidence graph (image-specific KG instance)│
          ▼                                                      ▼
 ╔════════════════════════════════════════════════════════════════════════╗
 ║                       PROPOSE  →  VERIFY  →  ARBITRATE                 ║
 ╠════════════════════════════════════════════════════════════════════════╣
 ║  PROPOSERS (each emits a candidate = ⟨answer, anchor, head_id, conf⟩):  ║
 ║    • E_perc-direct : answer from perception           anchor = regions ║
 ║    • E_sym         : answer from symbolic query       anchor = subgraph║
 ║                       (operates on LINKED FACTS, never pixels)         ║
 ║    • E_rag         : answer from similar cases/report anchor = passages║
 ║                       (INDEPENDENT of E_perc findings — the hedge)     ║
 ║                                                                        ║
 ║  VERIFIER (the engine, expert-agnostic) — for each candidate:          ║
 ║    run subsumption / disjointness / laterality / anatomy checks on     ║
 ║    (perceptual facts ∪ ontology axioms)                                ║
 ║      → PASS  (+ engine derivation subgraph = the faithful trace)       ║
 ║      → FAIL  (+ violated axiom)                                        ║
 ║      → UNVERIFIABLE (question outside ontology's expressive coverage)  ║
 ║                                                                        ║
 ║  ARBITRATION:                                                          ║
 ║    1. PASS & agree            → emit answer + engine derivation (corrob.)║
 ║    2. PASS & conflict         → engine derivation wins; else            ║
 ║                                 faithfulness-priority among verified     ║
 ║                                 (E_sym > E_rag > E_perc)                ║
 ║       (E_perc⊥E_rag disagree  → lean ABSTAIN/flag: two independent      ║
 ║        on a perceptual fact     perceptual paths conflict ⇒ low trust)  ║
 ║    3. none verifiable, high   → emit E_perc flagged "perception-only,   ║
 ║       perception confidence     NOT symbolically verified" (lower tier) ║
 ║    4. nothing passes, low conf→ ABSTAIN (→ selective-prediction)        ║
 ╚════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
                       Answer + Faithful Trace + Tier
                  (LLM renders engine derivation into prose;
                   adds no new inference)
```

### Role of the trained router (demoted, but kept)

The router is **no longer the arbiter**. It is a small MLP that:
- **predicts which head is likely to PASS verification first**, so we avoid running all three on every
  question (cost control); and
- **breaks ties** among candidates that the engine verifies as equally valid.

It is still *faithfulness-supervised* — its label is "the head whose candidate **passed verification
AND was correct**" — but a wrong router decision now degrades *efficiency*, not *faithfulness*, because
the engine still verifies whatever is proposed.

---

## 5. The symbolic engine

### 5.0 What the engine *is* (one definition, two roles)
The engine is a **deterministic, query-typed graph-reasoning procedure** over a static knowledge base
`K = (perceptual facts ∪ ontology axioms)`:

```
Engine(Q, K) → ⟨verdict, π⟩      verdict ∈ {PASS, FAIL, UNVERIFIABLE},  π ⊆ K (derivation subgraph)
```

It is not a net and has no learned weights — same `K` + same `Q` ⇒ same output. The derivation `π` is a
**byproduct** of the search that produced the verdict (this is what makes faithfulness by-construction).
`E_sym` and the verifier are **the same code in two modes**: `E_sym = Engine(Q, K)` ("find an answer");
`verify(c) = Engine(query_from_candidate(c), K)` ("check a candidate"). Because the engine owns the
closure condition in both modes, **completeness is the engine's responsibility, not the proposer's**.

### 5.1 Knowledge substrate
A **small, curated, frozen ontology DAG** (~hundreds of nodes), NOT full RadLex/SNOMED. Covers the
dataset's finding set + a few abstraction layers + only the relations reasoning actually uses:
`is-a`, `part-of`, `disjoint-with`, `laterality`, anatomy hierarchy. Version-controlled.

**Why not RadLex/SNOMED.** They are *terminologies*, not reasoning bases. RadLex (~68k terms) gives an
`is-a` hierarchy and synonyms but has **no `disjoint-with`, no closed-world exclusion lists, no
laterality composition** — it lists *what exists*, not *what follows from what*. SNOMED CT (~350k) adds
relations but targets clinical coding, and is too large/general (noise > signal) for CXR reasoning.
KRONOS seeds its `is-a` skeleton from RadLex, then **adds** the three relations they lack and **cuts**
the tens of thousands of irrelevant nodes. On the VinDr pilot the label set is closed (~22 findings), so
concept-linking is a lookup and DAG curation is days of work, not a research problem.

### 5.2 Four reasoning roles ontology provides
- **Subsumption (is-a):** lift fine labels to asked concept ("cardiomegaly" ⊢ "cardiac abnormality").
- **Disjointness / mutual exclusion:** consistency + absence answers.
- **Anatomy mapping:** pixel region → anatomical name (answers *Where*).
- **Attribute composition:** laterality / morphology composition (left/right, focal/diffuse).

### 5.3 Termination = engine-owned closure conditions (by question type)
| Question type | Monotonicity | Closure condition (engine decides "sufficient") |
|---|---|---|
| Existential / subsumption (`∃ finding is-a* X`) | monotone | early-exit safe: one witness `is-a* X` suffices |
| Negation / absence (`¬∃ ...`) | **non-monotone** | **closed-world enumeration**: every item on the static curated exclusion list (disjoint + subtype + confusable findings) must be checked before concluding absence |
| Relational / multi-hop (open) | — | no declarative closure → **budget + principled abstain** |

**The entire completeness risk is concentrated in the negation row.** Existential questions are already
safe; open multi-hop falls back to abstention. So the only hand-curation burden that affects soundness
is the per-finding closed-world exclusion list — keep it static, self-controlled, and version-tracked.

**Question-type is safety-critical and parsed conservatively.** The `type` field selects the closure
condition, so misclassifying a negation as existential could trigger an unsound early-exit. The type is
a constrained field of the parse (surface cues: "is there" → ∃, "clear/no/rule out" → ¬∃, "where/which
side" → relational). Rule: **on ambiguity, default to the *stronger* closure** (closed-world or abstain),
never to early-exit. Report type-classification accuracy as a separate, measurable error source.

### 5.4 Abstention
When ontology coverage cannot express the question, or closure is unreachable within budget, E_sym
returns **abstain** rather than guessing. The arbitrator then relies on E_rag / E_perc (lower
faithfulness tier) or abstains globally. Report abstention rate as *coverage*.

---

## 6. Why the faithfulness claim survives review

- **Soundness-faithfulness:** the answer provably follows from cited facts (engine-derived, not
  LLM-narrated). Trajectory freedom of any "where to look" proposer is harmless because EF scores the
  *derivation*, not the path.
- **Completeness:** patched exactly where it breaks (negation) via closed-world enumeration owned by
  the engine, not the proposer.
- **Causality:** the deletion/perturbation test shows the trace is load-bearing.
- **Coverage honesty:** abstention + reported coverage prevents "winning EF by abstaining" from being
  hidden.
- **No fusion contamination:** a single attributable derivation is always emitted; corroboration by
  multiple verified heads *raises confidence* without blending traces.
- **Perception-error honesty (the residual risk):** the engine cannot fix a wrong fact — garbage in,
  garbage out. What the design buys instead of perfect perception: (a) an **independent** second
  opinion (`E_rag`) that surfaces *conflict* rather than silently following a bad reading; (b)
  consistency checks that catch *internally inconsistent* hallucinations; (c) confidence-thresholded
  **abstention** that converts uncertain errors into safe non-answers; (d) bbox + trace that make the
  error **human-auditable**. A confident, consistent, single error that `E_rag` also misses still
  propagates — reported honestly via `linking_accuracy` and head-complementarity, not hidden.

### 6.1 Positioning vs the nearest prior work
- **vs LLM chain-of-thought** (unfaithful: ~13% implicit post-hoc rationalization, Arcuschin 2025;
  Med-VLM explanations survive image perturbation, Huang 2025) — KRONOS does not *narrate* a trace, it
  *derives* one; the deletion test proves it is load-bearing.
- **vs MedReason** (Wang 2025): MedReason uses a KG **at training time** to synthesize reasoning data,
  then fine-tunes an LLM — at inference the KG is gone and the LLM self-generates. KRONOS is
  **KG-at-inference**: the engine walks the DAG live, so the trace cannot drift from the graph.
- **vs generate-then-verify** (formal-methods lineage, e.g. VERGE 2025): same decoupling of a fallible
  proposer from a sound verifier — KRONOS applies it to multimodal medical VQA with a graph verifier.

---

## 7. Datasets & roadmap

| Stage | Dataset | Why | Watch-outs |
|---|---|---|---|
| Pilot | **VinDr-CXR-VQA** | closed label set → concept-linking is a lookup; bbox gold for EF | small scope |
| Validate negation | **PadChest-GR** | explicit *absent-finding* annotations → directly tests closed-world enumeration; openly accessible | report-gen oriented, small; use as targeted negation test, not main set |
| Scale | **GEMeX** (ICCV'25) | large; radiologist-refined region gold + reasoning → **keeps EF@k computable at scale**; ImaGenome anatomy grounding fits the anatomy role | gold *reasoning text* is GPT-4o-synthetic → score EF on **region gold only**; MIMIC/ImaGenome lineage → secure PhysioNet credentialing early |

Recommended order: VinDr → PadChest-GR (cheap negation validation) → GEMeX (scale). Scaling vocabulary
means scaling the curated DAG **and** the closed-world lists — budget curation as a first-class task;
lagging curation reopens completeness holes, so track per-finding closed-world coverage %.

---

## 8. Evaluation plan

**Metrics**
- **EF@k** on the derivation subgraph vs gold evidence regions (faithfulness).
- **Risk–coverage / selective-accuracy curves** (answer-when-faithful + abstain) — the primary lens,
  *not* raw top-1 accuracy.
- Task accuracy / F1 (for comparability with prior work).
- `linking_accuracy`, ontology `coverage %`, abstention rate (reported, not hidden).

**Headline ablations (acceptance hinges on these)**
1. Faithfulness-supervised vs accuracy-supervised router → EF must improve.
2. Full system vs **dense-retrieval-flat** baseline → must beat it (else ontology-as-structure is just cost).
3. Propose-then-verify vs single best head, and vs naive fusion → show selection-by-verification
   wins on faithfulness *at matched coverage*.

**Required diagnostics (run BEFORE writing)**
- **Head complementarity:** % of questions where *exactly one* head is correct. If E_sym and E_perc
  agree ~95% of the time, the routing/verification story is weak — measure first. (Keeping E_rag
  *independent* is what creates the complementarity that justifies the core.)
- **Reasoning-need statistics:** % of questions that genuinely require subsumption/relational reasoning
  vs pure perception. Justifies the engine's existence.
- **Deletion test:** answer flips when a trace fact/edge is removed → trace is causal.
- **Trivial selection baselines** for the router (max-confidence, fixed priority).

---

## 9. Cost & reproducibility

Frozen vision/LLM backbones. Engine = CPU inference over a precomputed DAG; marginal cost of E_sym ≈ a
graph traversal on top of perception. Only the small router is trained. No per-question multi-call
planner loop (the discarded E-SyM design); proposers run at most once, ordered by the router for cost.

---

## 10. Open risks — what actually decides A* acceptance

The architecture is coherent and defensible; acceptance depends on **empirical results**, not design
elegance. The deciding risks:

1. **Complementarity may be low.** If heads rarely disagree, propose-then-verify adds little over a
   single head. *Measure before committing.*
2. **EF must capture causality, not overlap.** If the deletion test is weak, reviewers reframe EF as
   "localization plausibility," not faithfulness — the exact flaw used to reject earlier designs.
3. **Must beat dense-retrieval-flat.** If structured reasoning doesn't beat a flat retriever on both
   accuracy and EF, the ontology is overhead.
4. **Curation scaling.** Completeness on negation depends on hand-curated exclusion lists keeping pace
   with vocabulary; report coverage honestly and abstain on the uncovered tail.
5. **Concept-linking is a correlated-error source.** E_perc errors propagate to E_sym; the independent
   E_rag hedge and a separately reported linking accuracy are the mitigations.

**Bottom line:** this is the strongest version of the idea, with the faithfulness hole closed and the
weak top-1 router replaced by verification-grounded selection. No one can promise A* before the four
headline experiments produce numbers — those are the gate.

---

## 11. Immediate next actions
1. Run the **reasoning-need** and **head-complementarity** diagnostics on VinDr-CXR-VQA.
2. Freeze the curated DAG relation set and the per-finding closed-world exclusion lists.
3. Implement the verifier (subsumption/disjointness/laterality/anatomy) as expert-agnostic checks.
4. Define EF@k formally on the derivation subgraph + specify the deletion-test protocol.
5. Stand up the dense-retrieval-flat baseline early.