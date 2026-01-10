import json

# Carica il JSON di input da file
with open("", "r", encoding="utf-8") as f:
    data = json.load(f)

# Prima operazione: sostituisce output vuoto con 1
for entry in data:
    if entry.get("output", None) == "":
        entry["output"] = 1

# Seconda operazione: filtra solo le entry con output == 1
filtered_data = [entry for entry in data if entry.get("output") == 1]

# Salva il nuovo JSON filtrato
with open("output_only_ones.json", "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, indent=4, ensure_ascii=False)

print("Operazione completata.")
