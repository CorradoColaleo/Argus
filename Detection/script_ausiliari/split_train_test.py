import json
import random
from collections import defaultdict

# --- Caricamento dei dataset ---
with open(r"C:\Users\corra\Desktop\università\AISE\progetto\Argus\dataset\dataset_unito_senza_metadati.json", "r", encoding="utf-8") as f:
    dataset1 = json.load(f)

with open(r"C:\Users\corra\Desktop\università\AISE\progetto\Argus\dataset\dataset_finale_con_metadati.json", "r", encoding="utf-8") as f:
    dataset2 = json.load(f)

# --- Funzione per rimuovere duplicati basata su 'text' ---
def remove_duplicates(dataset):
    seen = set()
    unique_dataset = []
    for item in dataset:
        if item["text"] not in seen:
            unique_dataset.append(item)
            seen.add(item["text"])
    return unique_dataset

dataset1 = remove_duplicates(dataset1)
dataset2 = remove_duplicates(dataset2)

print(f"[DEBUG] Dataset1: {len(dataset1)} campioni dopo rimozione duplicati")
print(f"[DEBUG] Dataset2: {len(dataset2)} campioni dopo rimozione duplicati\n")

# --- Funzione per separare per label ---
def split_by_label(dataset):
    label_dict = defaultdict(list)
    for item in dataset:
        label_dict[item["label"]].append(item)
    return label_dict

dataset1_by_label = split_by_label(dataset1)
dataset2_by_label = split_by_label(dataset2)

for i, label_dict in enumerate([dataset1_by_label, dataset2_by_label], 1):
    print(f"[DEBUG] Dataset{i} per label:")
    for label, items in label_dict.items():
        print(f"  Label {label}: {len(items)} campioni")
    print()

# --- Calcolo numero di campioni per il train ---
# Prendiamo metà dei campioni da ogni dataset per il train
train_size_per_dataset = min(len(dataset1), len(dataset2)) // 2
print(f"[DEBUG] Numero di campioni da prendere per train per ciascun dataset: {train_size_per_dataset}\n")

def sample_balanced(label_dict, num_samples):
    half = num_samples // 2
    sampled = []
    for label in [0, 1]:
        label_items = label_dict.get(label, [])
        if len(label_items) < half:
            raise ValueError(f"Non ci sono abbastanza campioni con label {label} per bilanciare il train.")
        selected = random.sample(label_items, half)
        sampled += selected
        print(f"[DEBUG] Estratti {len(selected)} campioni con label {label}")
    return sampled

# --- Creazione train dataset ---
train_dataset = []
train_dataset += sample_balanced(dataset1_by_label, train_size_per_dataset)
train_dataset += sample_balanced(dataset2_by_label, train_size_per_dataset)

print(f"\n[DEBUG] Train dataset totale: {len(train_dataset)} campioni")
print(f"[DEBUG] Esempi primi 5 campioni train:")
for item in train_dataset[:5]:
    print(f"  Label: {item['label']}, Text: {item['text'][:50]}...")

# --- Creazione test dataset ---
train_texts = set(item["text"] for item in train_dataset)

test_dataset = []
for dataset in [dataset1, dataset2]:
    for item in dataset:
        if item["text"] not in train_texts:
            test_dataset.append(item)

# Rimuovere eventuali duplicati nel test
test_dataset = remove_duplicates(test_dataset)

print(f"\n[DEBUG] Test dataset totale: {len(test_dataset)} campioni")
print(f"[DEBUG] Esempi primi 5 campioni test:")
for item in test_dataset[:5]:
    print(f"  Label: {item['label']}, Text: {item['text'][:50]}...")

# --- Salvataggio dei dataset ---
with open(r"C:\Users\corra\Desktop\università\AISE\progetto\Argus\dataset\train_dataset.json", "w") as f:
    json.dump(train_dataset, f, indent=4)

with open(r"C:\Users\corra\Desktop\università\AISE\progetto\Argus\dataset\test_dataset.json", "w") as f:
    json.dump(test_dataset, f, indent=4)

print(f"Train dataset: {len(train_dataset)} campioni")
print(f"Test dataset: {len(test_dataset)} campioni")
