import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_chat_model_cpu(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path).to("cpu")
    model.eval()
    return tokenizer, model

# === Load belief memory
with open(r"C:\Users\kuzey\OneDrive\Masaüstü\CurleysWife\data\belief.txt", "r", encoding="utf-8") as f:
    beliefs = f.read().strip()

# === Global memory
chat_history = []

def safe_chat_cpu(user_input, tokenizer, model, max_tokens=100):
    global chat_history

    # === Build prompt
    prompt = "[Memory]\n" + beliefs + "\n\n[Chat History]\n"
    for turn in chat_history:
        prompt += f"User: {turn['user']}\nCurley's Wife: {turn['bot']}\n"
    prompt += f"User: {user_input}\nCurley's Wife:"

    # === Tokenize with safe truncation
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024 - max_tokens)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.75,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )

    # === Decode only new response
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    reply = decoded.split("Curley's Wife:")[-1].strip().split("\n")[0]

    chat_history.append({"user": user_input, "bot": reply})
    return reply

import os
import tkinter as tk
from tkinter import scrolledtext, filedialog
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pyttsx3

# === SETTINGS ===
MODEL_DIRS = {
    "GPT2-Medium": r"C:\Users\kuzey\OneDrive\Masaüstü\CurleysWife\model\curleyswife_gpt2medium",
    "DistilGPT2": r"C:\Users\kuzey\OneDrive\Masaüstü\CurleysWife\model\curleyswife_distilgpt2"
}
BELIEF_PATH = r"C:\Users\kuzey\OneDrive\Masaüstü\CurleysWife\data\belief.txt"

# === LOAD BELIEF ===
with open(BELIEF_PATH, "r", encoding="utf-8") as f:
    BELIEF = f.read().strip()

# === TTS ===
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

# === MODEL LOADING ===
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return tokenizer, model

# === INIT DEFAULT MODEL ===
current_tokenizer, current_model = load_model(MODEL_DIRS["GPT2-Medium"])
history = []

# === CHAT FUNCTION ===
def chat(user_input, tokenizer, model):
    global history
    prompt = "[Memory]\n" + BELIEF + "\n\n[Chat History]\n"
    for msg in history:
        prompt += f"User: {msg['user']}\nCurley's Wife: {msg['bot']}\n"
    prompt += f"User: {user_input}\nCurley's Wife:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.75,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply = decoded.split("Curley's Wife:")[-1].strip().split("\n")[0]
    history.append({"user": user_input, "bot": reply})
    return reply

# === UI FUNCTIONS ===
def send_message():
    user_input = entry.get()
    if not user_input.strip():
        return

    root.title("Thinking...")

    response = chat(user_input, current_tokenizer, current_model)

    root.title("Chat with Curley's Wife")

    chat_window.insert(tk.END, "You: " + user_input + "\n", "user")
    chat_window.insert(tk.END, "Curley's Wife: " + response + "\n\n", "bot")
    chat_window.see(tk.END)
    speak(response)
    entry.delete(0, tk.END)

def change_model(new_model_name):
    global current_tokenizer, current_model, history
    current_tokenizer, current_model = load_model(MODEL_DIRS[new_model_name])
    history = []
    chat_window.insert(tk.END, f"\n🔁 Switched to **{new_model_name}**\n\n", "switch")

# === TKINTER UI ===
root = tk.Tk()
root.title("Chat with Curley's Wife")

frame = tk.Frame(root, bg="#222")
frame.pack(padx=10, pady=10)

model_switch = tk.StringVar(root)
model_switch.set("GPT2-Medium")
model_menu = tk.OptionMenu(frame, model_switch, *MODEL_DIRS.keys(), command=change_model)
model_menu.config(bg="gray20", fg="white")
model_menu.grid(row=0, column=0, padx=5, pady=5)

chat_window = scrolledtext.ScrolledText(frame, width=70, height=22, bg="#1e1e1e", fg="white", font=("Courier", 10))
chat_window.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

# === Text Tag Styling ===
chat_window.tag_config("user", foreground="#90ee90", font=("Courier", 10, "bold"))  # light green
chat_window.tag_config("bot", foreground="#ff99cc", font=("Courier", 10))           # soft pink
chat_window.tag_config("switch", foreground="#00bfff", font=("Courier", 10, "bold"))# cyan

entry = tk.Entry(frame, width=50, font=("Courier", 10))
entry.grid(row=2, column=0, padx=5, pady=5)

send_button = tk.Button(frame, text="Send", command=send_message, bg="#444", fg="white")
send_button.grid(row=2, column=1, padx=5)

root.mainloop()
