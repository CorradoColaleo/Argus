import csv
import json

CSV_INPUT = "dataset.csv"
JSON_OUTPUT = "dataset.json"

INSTRUCTION = (
    'You are a classification model specializing in emails, and your job is to detect phishing: '
    'respond only with "0" if it is not phishing or "1" if it is phishing, '
    'without explanations, symbols, additional letters, or other characters.'
)

output = []

with open(CSV_INPUT, newline="", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        email_text = row["Email Text"].strip()
        email_type = row["Email Type"].strip()

        if email_type == "Safe Mail":
            label = 0
        elif email_type == "Phishing Email":
            label = 1
        else:
            # opzionale: salta righe non valide
            continue

        output.append({
            "instruction": INSTRUCTION,
            "text": email_text,
            "label": label
        })

with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)

print(f"Creato file {JSON_OUTPUT} con {len(output)} entry.")
