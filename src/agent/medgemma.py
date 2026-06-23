"""MedGemma 4B agent — implements Agent Protocol for tree search.

Frozen, prompt-only, greedy decoding. Uses google/medgemma-4b-it via
HuggingFace transformers (AutoModelForImageTextToText).
When load_model=False, model is not loaded (for unit tests with mocked inference).
"""
from src.agent.prompt import build_prompt, parse_output


class MedGemmaAgent:
    def __init__(self, model_path="weights/medgemma-4b-it", image=None,
                 quantize=False, load_model=True):
        self._model = None
        self._processor = None
        self._image = image
        self._inference_fn = None

        if load_model and model_path:
            self._load(model_path, quantize)

    def _load(self, model_path, quantize):
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
            import torch

            kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
            if quantize:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )

            self._processor = AutoProcessor.from_pretrained(model_path)
            self._model = AutoModelForImageTextToText.from_pretrained(
                model_path, **kwargs
            )
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
        import torch

        # Gemma3 needs the chat template to place the <start_of_image> token;
        # passing raw text + image to the processor fails ("0 image tokens").
        content = [{"type": "text", "text": prompt}]
        if self._image is not None:
            content.insert(0, {"type": "image", "image": self._image})
        messages = [{"role": "user", "content": content}]

        inputs = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self._model.device)

        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs, max_new_tokens=256, do_sample=False,
            )
        return self._processor.decode(outputs[0][input_len:], skip_special_tokens=True)
