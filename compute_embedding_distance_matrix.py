import json
import torch
from safetensors import safe_open
from convert_lang_code import LANG_CODES

WORK = "tracked_work/joseph"
VECTORS_FILE = "embedding_vectors.json"

def finetune_dir(code):
    suffix = "_v2" if code == "sqf" else ""
    return f"{WORK}/fb_new3_qvoud_vecload_{code}_nllb{suffix}"

# Load pre-saved vectors
try:
    with open(VECTORS_FILE) as f:
        saved = {code: torch.tensor(vec, dtype=torch.float32) for code, vec in json.load(f).items()}
    print(f"Loaded {len(saved)} pre-saved embeddings from {VECTORS_FILE}")
except FileNotFoundError:
    saved = {}

missing = []
embeddings = {}

for code in LANG_CODES:
    path = f"{finetune_dir(code)}/final/adapter_model.safetensors"
    try:
        f = safe_open(path, framework="pt")
        fresh = f.get_tensor("base_model.model.model.shared.modules_to_save.weight")[-1].to(torch.float32)
    except FileNotFoundError:
        fresh = None

    if fresh is not None and code in saved:
        print(f"\nConflict for '{code}': both a finetune run and a pre-saved embedding exist.")
        while True:
            choice = input("  Use (f)inetune or (s)aved? [f/s]: ").strip().lower()
            if choice in ("f", "s"):
                break
        embeddings[code] = fresh if choice == "f" else saved[code]
    elif fresh is not None:
        embeddings[code] = fresh
    elif code in saved:
        embeddings[code] = saved[code]
    else:
        missing.append(code)
        print(f"MISSING: {code}")

if missing:
    print(f"\nSkipped {len(missing)} missing runs: {missing}")

# Save vectors
with open(VECTORS_FILE, "w") as f:
    json.dump({code: vec.tolist() for code, vec in embeddings.items()}, f, indent=2)
print(f"\nEmbedding vectors saved to {VECTORS_FILE}")

# Pairwise L2 distances
present = [c for c in LANG_CODES if c not in missing]
distances = {c: {} for c in present}
for i, c1 in enumerate(present):
    for c2 in present[i+1:]:
        d = (embeddings[c1] - embeddings[c2]).norm().item()
        distances[c1][c2] = d
        distances[c2][c1] = d
    distances[c1][c1] = 0.0

with open("embedding_distances.json", "w") as f:
    json.dump(distances, f, indent=2)

print(f"Distance matrix for {len(present)} languages saved to embedding_distances.json")
print("Languages:", present)
