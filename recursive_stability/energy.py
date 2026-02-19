# energy.py
from sentence_transformers import SentenceTransformer, util

from .config import EMBEDDING_MODEL_NAME

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def compute_energy(prompt, current, previous=None):
    model = get_model()
    p_emb = model.encode(prompt, convert_to_tensor=True)
    c_emb = model.encode(current, convert_to_tensor=True)

    grounding = 1 - util.cos_sim(c_emb, p_emb).item()

    stability = 0
    if previous:
        prev_emb = model.encode(previous, convert_to_tensor=True)
        stability = 1 - util.cos_sim(c_emb, prev_emb).item()

    return grounding + stability, grounding, stability

def foreign_char_ratio(text):
    if not text:
        return 0
    return sum(1 for c in text if ord(c) > 127) / len(text)
