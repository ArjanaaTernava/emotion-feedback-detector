import torch
import json
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Paths to the trained model and label mapping
MODEL_DIR = "../../models/preprocess-3-epoka-bert-model-80-20-new-dataset/model"
LABEL_PATH = "../../models/preprocess-3-epoka-bert-model-80-20-new-dataset/label_mapping.json"

# Load label mapping and create id-to-label dictionary
with open(LABEL_PATH, "r") as f:
    label_map = json.load(f)
id2label = {v: k for k, v in label_map.items()}

print("🔄 Loading trained model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

MAX_LEN = 128
EMOJI = {
    "sadness": "😢",
    "joy": "😊",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
    "neutral": "😐"
}

# Function to run emotion prediction for a given text
def predict_ui(text):
    if not text.strip():
        return "⚠️ Please enter some text."

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)
    label_id = int(torch.argmax(probs))
    confidence = probs[0][label_id].item()
    label = id2label[label_id]
    emoji = EMOJI.get(label, "💬")

    details = "\n".join(
        f"{id2label[i]}: {probs[0][i]:.2f}"
        for i in range(len(id2label))
    )

    return f"""
## {emoji} Emotion: **{label.upper()}**
### Confidence: **{confidence:.2%}**

---  
### 🔍 All Emotion Scores:
{details}
"""

# Build Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# 🎭 Emotion Classification Chat")
    gr.Markdown("Type a sentence and the AI will analyze its emotion.")

    textbox = gr.Textbox(
        lines=4,
        placeholder="Write your sentence here...",
        label="Your message"
    )

    output = gr.Markdown()

    button = gr.Button("Analyze Emotion", variant="primary")

    # Link button click to prediction function
    button.click(fn=predict_ui, inputs=textbox, outputs=output)

# Launch the Gradio app
app.launch()