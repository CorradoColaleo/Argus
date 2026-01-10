import json
import ollama

INPUT_JSON = "dataset_generazione.json"
OUTPUT_JSON = "dataset_generazione_with_prompts.json"

# Cambiato in Llama 3.1 per migliori prestazioni di ragionamento
MODEL_NAME = "llama3.1:70b" 

# Carica dataset
try:
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Errore: Il file {INPUT_JSON} non esiste.")
    exit()

total = len(data)

for i, entry in enumerate(data, start=1):
    print(f"Processing entry {i}/{total}...", end="\r")

    email_text = entry.get("input", "").strip()
    if not email_text:
        entry["output"] = ""
        continue

    # Prompt ottimizzato per Llama 3.1
    # L'uso dei delimitatori aiuta il modello a non confondersi tra istruzioni e testo della mail
    prompt_sistema = f"""### TASK:
Analyze the provided phishing email and infer a plausible USER PROMPT that could have been used to generate it.

RULES for the output:

Ensure the opening phrase is meaningfully different in wording and intent from the openings used in previous outputs;

Clearly indicate which brand, service, or organization is being impersonated.

Describe the overall scenario or justification used in the email in a way that reflects how a human would naturally frame the situation, rather than following a rigid or templated structure.

Indicate what the recipient is prompted to do, allowing flexibility in phrasing and emphasis.

Reflect any urgency, pressure, deadline, or implied consequence present in the email, if applicable.

The inferred prompt should encourage emails that sound human-written, with natural variation in structure, flow, and emphasis across different entries.

Do not include explanations, meta-comments, or quotation marks.

Output only the inferred prompt text itself.

Do not mirror the structure, wording, or clause order of any prior examples.

Allow variation in tone, level of detail, and sentence structure across different outputs

### EMAIL TO ANALYZE:
{email_text}

### YOUR INFERRED USER PROMPT:

"""

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt_sistema}],
            options={
                "temperature": 1.2, 
                "top_p": 0.9,
                "num_ctx": 4096      # Spazio di memoria per mail lunghe
            }
        )

        entry["output"] = response["message"]["content"].strip()

    except Exception as e:
        print(f"\nErrore alla entry {i}: {e}")
        entry["output"] = ""

    # Salva ogni 10 entry per non perdere i progressi senza rallentare troppo
    if i % 100 == 0 or i == total:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

print(f"\n\nCompletato! Risultati salvati in: {OUTPUT_JSON}")