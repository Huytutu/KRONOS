"""Frozen BiomedCLIP image encoder — lazy imports, GPU optional."""
import numpy as np

DEFAULT_MODEL_PATH = "weights/BiomedCLIP"


class CXREncoder:
    def __init__(self, model, preprocess, device):
        self._model = model
        self._preprocess = preprocess
        self._device = device

    def encode(self, image):
        import torch
        img_tensor = self._preprocess(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            emb = self._model.encode_image(img_tensor)
        emb = emb.cpu().numpy().flatten().astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        return emb


def load_encoder(model_path=DEFAULT_MODEL_PATH, device="cuda"):
    import torch
    import open_clip

    model, preprocess = open_clip.create_model_from_pretrained(
        "hf-hub:" + model_path, device=device,
    )
    model.eval()
    return CXREncoder(model, preprocess, device)
