"""LLaVA-Med 1.5 agent — implements Agent Protocol for tree search.

Frozen, prompt-only, greedy decoding. Supports fp16 and 4-bit quantization.
When load_model=False, model is not loaded (for unit tests with mocked inference).
"""
import json
import re
from src.contracts import Action, TreeNode, Query


TOOL_DESCRIPTIONS = """Available tools:
- is_a(node, target): check if node is-a target on ontology (returns path or None)
- disjoint(a, b): check if a and b are mutually exclusive (returns bool)
- anatomy_of(bbox): map bounding box to anatomy zone (returns zone name)
- compose_laterality(bbox): determine left/right/bilateral/midline from bbox position
- get_exclusion_list(name): get closed-world exclusion list for negation check
- retrieve(image_emb, k): retrieve top-k similar cases
- inspect(bbox): zoom into image region, describe finding (visual)
- re_detect(bbox): run detector on sub-region to find missed findings (visual)
- compare(bbox1, bbox2): compare two image regions (visual)
"""


class LLaVAMedAgent:
    def __init__(self, model_path=None, image=None, quantize=False, load_model=True):
        self._model = None
        self._processor = None
        self._image = image
        self._inference_fn = None

        if load_model and model_path:
            self._load(model_path, quantize)

    def _load(self, model_path, quantize):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
            if quantize:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )

            self._model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
            self._processor = AutoTokenizer.from_pretrained(model_path)
        except ImportError:
            pass

    def set_image(self, image):
        self._image = image

    def propose_actions(self, node, query, k=3):
        prompt = self.build_prompt(node, query, k)

        if self._inference_fn:
            raw = self._inference_fn(prompt, self._image)
        elif self._model:
            raw = self._run_model(prompt)
        else:
            return []

        return self.parse_output(raw)

    def build_prompt(self, node, query, k=3):
        parts = []

        parts.append(f"Question: {query.raw_question}")
        parts.append(f"Question type: {query.type}")
        if query.target:
            parts.append(f"Target: {query.target}")

        parts.append("\nEvidence facts:")
        if node.state_facts:
            for f in node.state_facts:
                parts.append(f"  - {f.concept} (conf={f.conf:.2f}, bbox={f.bbox}, lat={f.laterality})")
        else:
            parts.append("  (no findings)")

        if node.history:
            parts.append("\nHistory:")
            for action, obs in node.history:
                parts.append(f"  Action: {action.tool}({action.args})")
                parts.append(f"  Result: {obs.result} (ok={obs.ok})")

        if node.reflection:
            parts.append(f"\nReflection from failed branch: {node.reflection}")

        parts.append(f"\n{TOOL_DESCRIPTIONS}")

        parts.append(f"\nOutput up to {k} actions as a JSON array, e.g.:")
        parts.append('[{"tool": "is_a", "args": {"node": "cardiomegaly", "target": "cardiac_abnormality"}}]')
        parts.append('Or if you have enough evidence: Answer[your answer here]')

        return "\n".join(parts)

    def parse_output(self, raw):
        if not raw or not isinstance(raw, str):
            return []
        raw = raw.strip()

        answer_match = re.search(r'Answer\[(.+?)\]', raw)
        if answer_match:
            return [answer_match.group(1)]

        # Try the whole string as JSON, then a bracketed substring.
        actions = _parse_action_array(raw)
        if actions:
            return actions
        json_match = re.search(r'\[.*\]', raw, re.DOTALL)
        if json_match:
            return _parse_action_array(json_match.group())
        return []

    def _run_model(self, prompt):
        if not self._model or not self._processor:
            return ""
        inputs = self._processor(prompt, return_tensors="pt").to(self._model.device)
        with __import__("torch").no_grad():
            outputs = self._model.generate(
                **inputs, max_new_tokens=256,
                temperature=1.0, do_sample=False,
            )
        return self._processor.decode(outputs[0], skip_special_tokens=True)


VISUAL_TOOLS = ("inspect", "re_detect", "compare")


def _parse_action_array(text):
    """Parse a JSON array of {tool, args} dicts into Action objects. [] on failure."""
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []

    actions = []
    for item in parsed:
        if isinstance(item, dict) and "tool" in item:
            kind = "visual" if item["tool"] in VISUAL_TOOLS else "symbolic"
            actions.append(Action(tool=item["tool"], args=item.get("args", {}), kind=kind))
    return actions
