import json
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import re

class JsonViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("JSON Viewer con Navigazione")
        self.geometry("1100x750")

        self.data = []
        self.index = 0

        self._build_ui()

    def _build_ui(self):
        top = tk.Frame(self)
        top.pack(fill=tk.X, padx=10, pady=5)

        tk.Button(top, text="Apri JSON", command=self.load_json).pack(side=tk.LEFT)

        self.idx_label = tk.Label(top, text="Index: - / -")
        self.idx_label.pack(side=tk.LEFT, padx=20)

        tk.Button(top, text="◀ Prev", command=self.prev_item).pack(side=tk.LEFT)
        tk.Button(top, text="Next ▶", command=self.next_item).pack(side=tk.LEFT)

        body = tk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.instruction = self._section(body, "Instruction")
        self.input_text = self._section(body, "Input")
        self.output_text = self._section(body, "Output")

    def _section(self, parent, title):
        frame = tk.LabelFrame(parent, text=title)
        frame.pack(fill=tk.BOTH, expand=True, pady=5)

        txt = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=8)
        txt.pack(fill=tk.BOTH, expand=True)
        txt.configure(state=tk.DISABLED)

        frame.text_widget = txt
        return frame

    def load_json(self):
        path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            self.index = 0
            self.show_item()
        except Exception as e:
            messagebox.showerror("Errore", str(e))

    def show_item(self):
        if not self.data:
            return
        item = self.data[self.index]
        self._set_text(self.instruction.text_widget, self._clean_text(item.get("instruction", "")))
        self._set_text(self.input_text.text_widget, self._clean_text(item.get("input", "")))
        self._set_text(self.output_text.text_widget, self._clean_text(item.get("output", "")))
        self.idx_label.config(text=f"Index: {self.index} / {len(self.data)-1}")

    def _clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _set_text(self, widget, text):
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.configure(state=tk.DISABLED)

    def prev_item(self):
        if self.index > 0:
            self.index -= 1
            self.show_item()

    def next_item(self):
        if self.index < len(self.data) - 1:
            self.index += 1
            self.show_item()

if __name__ == "__main__":
    app = JsonViewer()
    app.mainloop()