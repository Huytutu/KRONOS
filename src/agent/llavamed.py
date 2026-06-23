"""LLaVA-Med 1.5 agent — implements Agent Protocol for tree search.

Frozen, prompt-only, greedy decoding. Supports fp16 and 4-bit quantization.
When load_model=False, model is not loaded (for unit tests with mocked inference).
"""
from src.agent.prompt import build_prompt, parse_output, SYSTEM_INSTRUCTION, TOOL_DESCRIPTIONS, VISUAL_TOOLS


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

            kwargs = {"torch_dtype": torch.float16, "device_map": "auto", "trust_remote_code": True}
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
        prompt = build_prompt(node, query, k)

        if self._inference_fn:
            raw = self._inference_fn(prompt, self._image)
        elif self._model:
            raw = self._run_model(prompt)
        else:
            return []

        return parse_output(raw)

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
