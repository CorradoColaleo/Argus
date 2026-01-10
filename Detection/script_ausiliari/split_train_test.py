import pandas as pd
from sklearn.model_selection import train_test_split


# 1. Caricamento del dataset
try:
    df = pd.read_json(r'C:\Users\franc\Downloads\dataset_detection_totale.json')
except ValueError:
    # Fallback per JSON Lines se il formato è quello
    df = pd.read_json(r'C:\Users\franc\Downloads\dataset_detection_totale.json', lines=True)

print(f"Totale campioni caricati: {len(df)}")
print(f"Distribuzione originale:\n{df['label'].value_counts()}")

# 2. Split del dataset
# test_size=0.25 -> Genera 25% test e 75% train
# stratify=df['label'] -> Mantiene la proporzione delle label (0 e 1) identica tra train e test
# random_state=42 -> Garantisce che lo split sia riproducibile 
train_df, test_df = train_test_split(
    df,
    test_size=0.25,
    stratify=df['label'],
    random_state=42,
    shuffle=True
)

# 3. Verifica del bilanciamento e delle dimensioni
print("\n--- DATASET DI TRAIN ---")
print(f"Dimensione: {len(train_df)} ({len(train_df)/len(df):.1%})")
print(f"Distribuzione label:\n{train_df['label'].value_counts()}")

print("\n--- DATASET DI TEST ---")
print(f"Dimensione: {len(test_df)} ({len(test_df)/len(df):.1%})")
print(f"Distribuzione label:\n{test_df['label'].value_counts()}")

# 4. Verifica Assenza Sovrapposizioni 
overlap = set(train_df.index).intersection(set(test_df.index))
if len(overlap) == 0:
    print("\nVerifica superata: Nessuna sovrapposizione tra Train e Test.")
else:
    print(f"\nATTENZIONE: Trovati {len(overlap)} campioni sovrapposti!")

# 5. Salvataggio dei file
train_df.to_json('train_dataset.json', orient='records', indent=4)
test_df.to_json('test_dataset.json', orient='records', indent=4)

print("\nFile 'train_dataset.json' e 'test_dataset.json' generati con successo.")