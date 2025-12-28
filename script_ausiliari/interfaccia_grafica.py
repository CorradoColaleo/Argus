import tkinter as tk
from tkinter import scrolledtext

# Dummy funzione per simulare inferenza
def analyze_email(email_body):
    return f"Analisi generata per la mail:\n{email_body[:50]}..."

# ----------------------------
# CREAZIONE INTERFACCIA AD ALTA DEFINIZIONE E RESPONSIVE
# ----------------------------

root = tk.Tk()
root.title("Email Security Analyzer")
root.geometry("1000x700")
root.configure(bg="#0f0f1a")

# Attiva scaling DPI (Windows/macOS)
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except:
    pass

# ----------------------------
# FRAME SUPERIORE: INPUT + PULSANTE
# ----------------------------
top_frame = tk.Frame(root, bg="#0f0f1a")
top_frame.pack(fill="x", padx=10, pady=10)

title = tk.Label(top_frame, text="Email Security Analyzer",
                 fg="#00ffaa", bg="#0f0f1a",
                 font=("Courier New", 36, "bold"))
title.pack(anchor="center", pady=10)

input_label = tk.Label(top_frame, text="Inserisci la mail:",
                       fg="#00ffea", bg="#0f0f1a",
                       font=("Courier New", 18))
input_label.pack(anchor="w")

input_box = scrolledtext.ScrolledText(top_frame, height=10, bg="#1a1a2e", fg="#00ffea",
                                      font=("Courier New", 16))
input_box.pack(fill="x", pady=5)

predict_btn = tk.Button(top_frame, text="Predict", bg="#00ffea", fg="#0f0f1a",
                        font=("Courier New", 18, "bold"),
                        command=lambda: on_predict())
predict_btn.pack(pady=5)

# ----------------------------
# FRAME INFERIORE: OUTPUT
# ----------------------------
bottom_frame = tk.Frame(root, bg="#0f0f1a")
bottom_frame.pack(fill="both", expand=True, padx=10, pady=10)

output_label = tk.Label(bottom_frame, text="Predizione del modello:",
                        fg="#00ffea", bg="#0f0f1a",
                        font=("Courier New", 18))
output_label.pack(anchor="w")

output_box = scrolledtext.ScrolledText(bottom_frame, bg="#1a1a2e", fg="#00ffea",
                                      font=("Courier New", 16))
output_box.pack(fill="both", expand=True, pady=5)
output_box.config(state=tk.DISABLED)

# ----------------------------
# FUNZIONE PREDICT
# ----------------------------
def on_predict():
    email_body = input_box.get("1.0", tk.END).strip()
    if email_body:
        output = analyze_email(email_body)
        output_box.config(state=tk.NORMAL)
        output_box.delete("1.0", tk.END)
        output_box.insert(tk.END, output)
        output_box.config(state=tk.DISABLED)

root.mainloop()
