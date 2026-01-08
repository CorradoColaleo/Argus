import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

NGRAM_SIZES = [7, 10, 15, 20]


def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()


def extract_body(input_field):
    match = re.search(r'body:\s*(.*)', input_field, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def build_ngrams(text, n):
    words = text.split(' ')
    return [
        ' '.join(words[i:i+n])
        for i in range(0, len(words), n)
        if len(words[i:i+n]) == n
    ]


def best_pair_for_ngrams(ngrams):
    if len(ngrams) < 2:
        return None, None, -1.0

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(ngrams)

    best_sim = -1.0
    best_i, best_j = None, None

    for i in range(len(ngrams)):
        for j in range(i + 1, len(ngrams)):
            sim = cosine_similarity(tfidf[i], tfidf[j])[0][0]
            if sim > best_sim:
                best_sim = sim
                best_i, best_j = i, j

    return ngrams[best_i], ngrams[best_j], best_sim


# ======================= MAIN =======================

json_path = "insert your path to the JSON file here"
output_path = "insert your path to the JSON file here"
threshold = float(input("Threshold di similarità (es. 0.60): "))

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

filtered_data = []         # <-- JSON finale senza duplicati
removed_count = 0
seen_count = 0

for idx, instance in enumerate(data, 1):
    seen_count += 1

    raw_input = instance.get("input", "")
    body_text = extract_body(raw_input)

    if not body_text:
        filtered_data.append(instance)
        continue

    cleaned = clean_text(body_text)

    global_best_sim = -1.0
    global_best_pair = (None, None)
    global_best_n = None

    for n in NGRAM_SIZES:
        ngrams = build_ngrams(cleaned, n)
        ng1, ng2, sim = best_pair_for_ngrams(ngrams)
        if sim > global_best_sim:
            global_best_sim = sim
            global_best_pair = (ng1, ng2)
            global_best_n = n

    # === DECISIONE DI RIMOZIONE ===
    if global_best_sim > threshold:
        removed_count += 1
        removed = True
    else:
        removed = False
        filtered_data.append(instance)  # <-- ISTANZA TENUTA

    # Preview prime 5
    if idx <= 5:
        print(f"\n=== ISTANZA {idx} PREVIEW ===")
        print(f"Dimensione n-grammi: {global_best_n}")
        print("\nN-gramma 1:")
        print(global_best_pair[0])
        print("\nN-gramma 2:")
        print(global_best_pair[1])
        print(f"\nSimilarità (coseno TF-IDF): {global_best_sim:.4f}")

    # Stampa metriche (prime 5 + ogni 500)
    if idx <= 5 or seen_count % 500 == 0:
        print(
            f"[{seen_count}] removed={removed} | "
            f"best_sim={global_best_sim:.4f} | "
            f"removed_total={removed_count}/{seen_count}"
        )

# ===== SALVATAGGIO JSON FILTRATO =====
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=2)

print("\n=== RISULTATO FINALE ===")
print(f"Istanze originali: {len(data)}")
print(f"Istanze rimosse: {removed_count}")
print(f"Istanze finali: {len(filtered_data)}")
print(f"Percentuale rimossa: {(removed_count / seen_count) * 100:.2f}%")
print(f"JSON filtrato salvato in: {output_path}")
