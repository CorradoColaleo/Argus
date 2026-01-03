# Questo script serve per controllare se ci sono sovrapposizioni tra i dataset di training e di testing
import json
import re
from collections import defaultdict

def normalize_text(text: str) -> str:
    """Normalizza il testo: minuscolo, spazi multipli → singolo spazio, rimuove spazi iniziali/finali."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

# Percorsi dei due dataset
training_path = r"C:\Users\corra\Desktop\università\AISE\progetto\Argus\dataset\dataset1.json"
testing_path  = r"C:\Users\corra\Desktop\università\AISE\progetto\Argus\dataset\dataset2.json"

# Carica i dataset
with open(training_path, "r", encoding="utf-8") as f:
    training_data = json.load(f)

with open(testing_path, "r", encoding="utf-8") as f:
    testing_data = json.load(f)

# Crea dizionario dei testi normalizzati del training
training_texts = defaultdict(list)
for idx, item in enumerate(training_data):
    norm_text = normalize_text(item["text"])
    training_texts[norm_text].append(idx)

# Controlla duplicati tra testing e training
duplicates = []
for test_idx, item in enumerate(testing_data):
    norm_text = normalize_text(item["text"])
    if norm_text in training_texts:
        duplicates.append({
            "training_indices": training_texts[norm_text],
            "testing_index": test_idx,
            "text": norm_text
        })

# Stampa risultati
if not duplicates:
    print("Nessun duplicato tra training e testing trovato.")
else:
    print(f"Trovati {len(duplicates)} duplicati tra training e testing:\n")
    for dup in duplicates:
        print(f"Indici training: {dup['training_indices']}, indice testing: {dup['testing_index']}")
        # Stampa testo troncato a 500 caratteri per comodità
        print(f"Testo duplicato (troncato 500 caratteri): {dup['text'][:500]}")
        print("-" * 120)
    print(f"Trovati {len(duplicates)} duplicati tra training e testing:\n")

