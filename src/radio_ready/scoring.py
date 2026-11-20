"""Similarity and toxicity scorers for filtering, with thresholds from ParaDeHate.

  Similarity  sentence-transformers/all-distilroberta-v1, cosine; threshold 0.70 (Sec. 3.2)
  Toxicity    unitary/unbiased-toxic-roberta, `toxicity` head; threshold 0.9 (Sec. 3.3)

Evaluation models (LaBSE, s-nlp/roberta_toxicity_classifier) are separate from these.
"""

import numpy as np
import torch

SIMILARITY_MODEL = "sentence-transformers/all-distilroberta-v1"
TOXICITY_MODEL = "unitary/unbiased-toxic-roberta"
SIMILARITY_THRESHOLD = 0.70  # ParaDeHate Sec. 3.2
TOXICITY_THRESHOLD = 0.9  # ParaDeHate Sec. 3.3


def device() -> str:
    name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda":
        print(f"scoring on GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("scoring on CPU (no CUDA device found)", flush=True)
    return name


class Similarity:
    def __init__(self, model: str = SIMILARITY_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model, device=device())

    def pairwise(self, a: list[str], b: list[str], batch_size: int = 64) -> np.ndarray:
        ea = self.model.encode(a, batch_size=batch_size, normalize_embeddings=True)
        eb = self.model.encode(b, batch_size=batch_size, normalize_embeddings=True)
        return (ea * eb).sum(axis=1)


class Toxicity:
    """Probability of the 'toxicity' head of unbiased-toxic-roberta (multi-label, sigmoid)."""

    def __init__(self, model: str = TOXICITY_MODEL):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).to(device()).eval()
        self.index = self.model.config.label2id["toxicity"]

    @torch.no_grad()
    def score(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = self.tokenizer(
                texts[i : i + batch_size], padding=True, truncation=True, max_length=256,
                return_tensors="pt",
            ).to(self.model.device)
            logits = self.model(**batch).logits[:, self.index]
            out.append(torch.sigmoid(logits).float().cpu().numpy())
        return np.concatenate(out) if out else np.array([])
