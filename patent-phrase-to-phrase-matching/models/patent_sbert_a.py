from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("AI-Growth-Lab/PatentSBERTa")


def get_sim(anchor: str, target: str) -> float:
    anchor_embed = model.encode([anchor])
    target_embed = model.encode([target])
    return float(1 - cosine(anchor_embed, target_embed))



examples = [
    ["renewable power", "renewable energy"],
    ["previously captured image", "image captured previously"],
    ["labeled ligand", "container labelling"],
    ["gold alloy", "platinum"],
    ["dissolve in glycol", "family gathering"],
]


if __name__ == '__main__':
    get_sim("renewable power", "renewable energy")
