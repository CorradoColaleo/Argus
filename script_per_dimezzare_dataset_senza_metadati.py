import json
import random
from collections import defaultdict, Counter

INPUT_FILE = "C:\\Users\\corra\\Desktop\\università\\AISE\\progetto\\dataset_sft.json"
OUTPUT_FILE = "C:\\Users\\corra\\Desktop\\università\\AISE\\progetto\\dataset_half_balanced.json"
SEED = 42

random.seed(SEED)

# -------------------------
# Caricamento dataset
# -------------------------
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# -------------------------
# Conteggio dataset originale
# -------------------------
original_counts = Counter(item["output"] for item in data)

print("=== Dataset originale ===")
print(f"Totale esempi: {len(data)}")
print(f"Output 0: {original_counts.get('0', 0)}")
print(f"Output 1: {original_counts.get('1', 0)}")
print()

# -------------------------
# Raggruppamento per classe
# -------------------------
by_label = defaultdict(list)
for item in data:
    by_label[item["output"]].append(item)

if "0" not in by_label or "1" not in by_label:
    raise ValueError("Il dataset deve contenere sia output '0' che '1'")

# -------------------------
# Calcolo dimensione ridotta
# -------------------------
total_target_size = len(data) // 2
per_class = total_target_size // 2

for label in ["0", "1"]:
    if len(by_label[label]) < per_class:
        raise ValueError(f"Esempi insufficienti per la classe {label}")

# -------------------------
# Campionamento bilanciato
# -------------------------
reduced_data = []
reduced_data.extend(random.sample(by_label["0"], per_class))
reduced_data.extend(random.sample(by_label["1"], per_class))

random.shuffle(reduced_data)

# -------------------------
# Conteggio dataset ridotto
# -------------------------
reduced_counts = Counter(item["output"] for item in reduced_data)

print("=== Dataset ridotto ===")
print(f"Totale esempi: {len(reduced_data)}")
print(f"Output 0: {reduced_counts.get('0', 0)}")
print(f"Output 1: {reduced_counts.get('1', 0)}")
print()

# -------------------------
# Salvataggio
# -------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(reduced_data, f, ensure_ascii=False, indent=2)

print(f"File salvato come: {OUTPUT_FILE}")