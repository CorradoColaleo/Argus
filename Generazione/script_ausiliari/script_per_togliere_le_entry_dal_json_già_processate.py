import json

# Carica il JSON di input da file
with open("C:\\Users\\corra\\Desktop\\università\\AISE\\progetto\\Argus\\Generazione\\script_ausiliari\\dataset_generazione_parziale_2_with_prompts.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Prima operazione: sostituisce output vuoto con 1
for entry in data:
    if entry.get("output", None) == "":
        entry["output"] = 1

# Salva tutte le entry con output != 1
output_not_ones = [entry for entry in data if entry.get("output") != 1]

with open("output_processed.json", "w", encoding="utf-8") as f:
    json.dump(output_not_ones, f, indent=4, ensure_ascii=False)

# Salva solo le entry con output == 1
output_only_ones = [entry for entry in data if entry.get("output") == 1]

with open("output_only_ones.json", "w", encoding="utf-8") as f:
    json.dump(output_only_ones, f, indent=4, ensure_ascii=False)

print("Operazione completata.")
