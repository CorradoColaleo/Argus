#Questo script serve solo per controllare se ci sono duplicati nei dataset di training e di testing
# Questo script serve per controllare se ci sono duplicati nei dataset di training e di testing
import json
import re
from collections import defaultdict

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)  # spazi multipli → singolo spazio
    text = text.strip()
    return text

# Apri il dataset
with open(r"C:\Users\corra\Desktop\università\AISE\progetto\Argus\dataset\train_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Mappa: text -> lista di indici
text_occurrences = defaultdict(list)

for idx, item in enumerate(data):
    text_occurrences[item["text"]].append(idx)

# Estrai solo i duplicati
duplicates = {text: idxs for text, idxs in text_occurrences.items() if len(idxs) > 1}

# Risultati
if not duplicates:
    print("Nessun duplicato esatto trovato nel campo 'text'.")
else:
    print(f"Trovati {len(duplicates)} testi duplicati:\n")
    for text, idxs in duplicates.items():
        print(f"Indici: {idxs}")
        print(f"Text (troncato a 300 caratteri): {text[:300]}")
        print("-" * 80)