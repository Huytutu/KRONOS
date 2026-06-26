"""Shared prompt construction, output parsing, and constants for all agents."""
import json
import re
from src.contracts import Action


SYSTEM_INSTRUCTION = """You are a medical VQA reasoning agent. You answer chest X-ray questions by calling tools, then emitting Answer[...] when you have enough evidence.

STRATEGY BY QUESTION TYPE:

existential ("Is there X?"):
  1. For each base fact, call is_a(fact, target) to check if it witnesses the target.
  2. If any returns a path → emit Answer[Yes].
  3. If none match and you suspect a missed finding → call re_detect(bbox) on a likely region, then retry is_a on new facts.
  4. If still no witness → emit Answer[No].

negation ("Is there NO X?" / "Are the lungs clear?"):
  1. Call get_exclusion_list(target) first.
  2. For each item in the list, check if any base fact matches via is_a.
  3. If ANY match is found → emit Answer[No] (finding IS present, so "no X" is false).
  4. If ALL items are checked and absent → emit Answer[Yes] (confirmed absent).

relational ("Where is X?" / "Which side?"):
  1. Find the fact matching the target.
  2. Call anatomy_of(bbox) or compose_laterality(bbox) on that fact's bbox.
  3. Emit Answer[result].

counting ("How many findings?"):
  1. Count distinct findings in the evidence facts.
  2. Emit Answer[count].

open ("Describe the main finding"):
  1. Summarize the evidence facts.
  2. Emit Answer[summary].

RULES:
- Prefer symbolic tools first (fast, deterministic). Use visual tools only when symbolic evidence is insufficient.
- Use re_detect only when base facts don't contain the target and you suspect the detector missed something.
- Never fabricate findings. Only report what tools return.
- Output up to k actions as a JSON array, OR Answer[your answer] if you have enough evidence.
"""

TOOL_DESCRIPTIONS = """Available tools:
- is_a(node, target): check if node is-a target on ontology (returns path or None)
- disjoint(a, b): check if a and b are mutually exclusive (returns bool)
- anatomy_of(bbox): map bounding box to anatomy zone (returns zone name)
- compose_laterality(bbox): determine left/right/bilateral/midline from bbox position
- get_exclusion_list(name): get closed-world exclusion list for negation check
- neighbors(node, direction): concepts linked on the may_cause graph; direction 'caused_by' (causes of node) or 'causes' (effects of node)
- find_path(source, target): shortest may_cause chain source -> target (multi-hop causal rationale), or None
- retrieve(image_emb, k): retrieve top-k similar cases
- inspect(bbox): zoom into image region, describe finding (visual)
- re_detect(bbox): run detector on sub-region to find missed findings (visual)
- compare(bbox1, bbox2): compare two image regions (visual)
"""

VISUAL_TOOLS = ("inspect", "re_detect", "compare")


def build_prompt(node, query, k=3):
    parts = [SYSTEM_INSTRUCTION]

    parts.append(f"Question: {query.raw_question}")
    parts.append(f"Question type: {query.type}")
    if query.target:
        parts.append(f"Target: {query.target}")
    elif query.type == "relational" and node.state_facts:
        parts.append("Target: (not specified — use the most prominent finding from evidence facts)")

    parts.append("\nEvidence facts:")
    if node.state_facts:
        for f in node.state_facts:
            parts.append(f"  - {f.concept} (conf={f.conf:.2f}, bbox={f.bbox}, lat={f.laterality})")
    else:
        parts.append("  (no findings detected)")

    if node.history:
        parts.append("\nActions taken so far:")
        for action, obs in node.history:
            parts.append(f"  Action: {action.tool}({action.args})")
            parts.append(f"  Result: {obs.result} (ok={obs.ok})")

    if node.reflection:
        parts.append(f"\nReflection from failed branch: {node.reflection}")

    parts.append(f"\n{TOOL_DESCRIPTIONS}")

    parts.append(f"\nRespond with up to {k} actions as a JSON array, e.g.:")
    parts.append('[{"tool": "is_a", "args": {"node": "cardiomegaly", "target": "cardiac_abnormality"}}]')
    parts.append('Or if you have enough evidence: Answer[your answer here]')

    if query.type == "open" and node.state_facts:
        parts.append("\nYou have detected findings listed above. Summarize them to answer the question. Emit Answer[your answer] directly.")

    return "\n".join(parts)


def parse_output(raw):
    if not raw or not isinstance(raw, str):
        return []
    raw = raw.strip()

    answer_match = re.search(r'Answer\[(.+?)\]', raw)
    if answer_match:
        return [answer_match.group(1)]

    actions = _parse_action_array(raw)
    if actions:
        return actions
    json_match = re.search(r'\[.*\]', raw, re.DOTALL)
    if json_match:
        return _parse_action_array(json_match.group())
    return []


VALID_TOOLS = {
    "is_a", "disjoint", "anatomy_of", "compose_laterality",
    "get_exclusion_list", "retrieve", "neighbors", "find_path",
    "inspect", "re_detect", "compare",
}


def _parse_action_array(text):
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []

    actions = []
    for item in parsed:
        if isinstance(item, dict) and "tool" in item and item["tool"] in VALID_TOOLS:
            kind = "visual" if item["tool"] in VISUAL_TOOLS else "symbolic"
            actions.append(Action(tool=item["tool"], args=item.get("args", {}), kind=kind))
    return actions
