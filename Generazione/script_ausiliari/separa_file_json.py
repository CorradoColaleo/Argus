import json

# File di input
input_file = "C:\\Users\\corra\\Downloads\\dataset_generazione.json"

# File di output
output_0_file = "C:\\Users\\corra\\Downloads\\genlabel0.json"
output_1_file = "C:\\Users\\corra\\Downloads\\genlabel1.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Separazione dei record
output_0 = [item for item in data if item.get("output") == 0]
output_1 = [item for item in data if item.get("output") == 1]

# Scrittura dei file
with open(output_0_file, "w", encoding="utf-8") as f:
    json.dump(output_0, f, ensure_ascii=False, indent=2)

with open(output_1_file, "w", encoding="utf-8") as f:
    json.dump(output_1, f, ensure_ascii=False, indent=2)

print(f"Creati {len(output_0)} record in {output_0_file}")
print(f"Creati {len(output_1)} record in {output_1_file}")
