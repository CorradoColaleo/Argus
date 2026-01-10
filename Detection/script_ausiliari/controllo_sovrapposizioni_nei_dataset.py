# Script per controllare sovrapposizioni ESATTE tra train e test
import json
from collections import defaultdict

# Percorsi dei dataset
training_path = r""
testing_path  = r""

# Caricamento dataset
with open(training_path, "r", encoding="utf-8") as f:
    training_data = json.load(f)

with open(testing_path, "r", encoding="utf-8") as f:
    testing_data = json.load(f)

# Costruisce dizionario dei testi del training (NESSUNA normalizzazione)
training_texts = defaultdict(list)
for idx, item in enumerate(training_data):
    training_texts[item["text"]].append(idx)

# Controllo sovrapposizioni ESATTE
duplicates = []
for test_idx, item in enumerate(testing_data):
    text = item["text"]
    if text in training_texts:
        duplicates.append({
            "training_indices": training_texts[text],
            "testing_index": test_idx,
            "text": text
        })

# Output
if not duplicates:
    print("Nessuna sovrapposizione ESATTA tra training e testing.")
else:
    print(f"Trovate {len(duplicates)} sovrapposizioni ESATTE tra training e testing:\n")
    for dup in duplicates:
        print(f"Indici training: {dup['training_indices']}, indice testing: {dup['testing_index']}")
        print(f"Testo duplicato (troncato 500 caratteri): {dup['text'][:500]}")
        print("-" * 120)
