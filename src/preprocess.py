"""
Saves:
- logs.txt
- metrics.json
- classification_report.txt
- confusion_matrix.png
- confusion_matrix.json
- loss_curve.png
- f1_curve.png
- label_mapping.json
- model + tokenizer (in model/ subfolder)
into the folder: preprocess-3-epochs-bert-model
"""

import os
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

import sys

class Tee:
    def __init__(self, filename):
        self.file = open(filename, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def main():

    DATA_PATH = "data/emotions-dataset.xlsx"  
    
    RUN_DIR = Path("preprocess-3-epochs-bert-model")
    RUN_DIR.mkdir(parents=True, exist_ok=True)


    # Log file
    log_path = RUN_DIR / "logs.txt"
    sys.stdout = Tee(log_path)
    print(f"Logging to: {log_path}")

    # Model & training config
    MODEL_NAME = "distilbert-base-uncased"
    MAX_LEN = 128
    EPOCHS = 3  
    LR = 2e-5
    TRAIN_BATCH = 16
    EVAL_BATCH = 32

    print("Loading dataset...")
    df = pd.read_excel(DATA_PATH)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("The file must contain 'text' and 'label' columns.")

    df = df[["text", "label"]].dropna()

    print("Sample rows:")
    print(df.head())

    emotion_labels = {
        "sadness": 0,
        "joy": 1,
        "love": 2,
        "anger": 3,
        "fear": 4,
        "surprise": 5,
    }

    # Save label mapping
    label_map_path = RUN_DIR / "label_mapping.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(emotion_labels, f, indent=2)
    print(f"💾 Saved label mapping → {label_map_path}")

    df["label"] = df["label"].map(emotion_labels)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    print(f"\nTrain size: {len(train_df)}, Test size: {len(test_df)}")

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(emotion_labels)
    )

    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN
        )

    print("Tokenizing...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    keep_cols = ["input_ids", "attention_mask", "label"]
    train_dataset = train_dataset.remove_columns(
        [c for c in train_dataset.column_names if c not in keep_cols]
    )
    test_dataset = test_dataset.remove_columns(
        [c for c in test_dataset.column_names if c not in keep_cols]
    )

    train_dataset.set_format("torch")
    test_dataset.set_format("torch")

    id2label = {v: k for k, v in emotion_labels.items()}

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average="macro")
        return {
            "accuracy": acc,
            "f1_macro": f1_macro,
        }

    print("Setting training arguments...")
    training_args = TrainingArguments(
        output_dir=str(RUN_DIR / "model"),   # model subfolder
        eval_strategy="epoch",               # using old-style arg for compatibility
        save_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=200,
        save_total_limit=2,
        report_to="none",
    )

    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.train()

    print("Final evaluation on test set...")
    eval_results = trainer.evaluate()
    print("\nEvaluation results:")
    print(eval_results)

    # Save metrics.json
    metrics_path = RUN_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Saved metrics → {metrics_path}")

    # Detailed classification report
    print("Generating classification report...")
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    true_labels = predictions.label_ids

    class_report = classification_report(
        true_labels,
        pred_labels,
        target_names=[id2label[i] for i in range(len(id2label))],
        digits=4
    )

    print("\nClassification report:")
    print(class_report)

    # Save classification report to file
    cr_path = RUN_DIR / "classification_report.txt"
    with open(cr_path, "w", encoding="utf-8") as f:
        f.write(class_report)
    print(f"💾 Saved classification report → {cr_path}")

    print("🧮 Computing confusion matrix...")
    cm = confusion_matrix(true_labels, pred_labels)
    cm_json = {
        id2label[i]: {
            "row": i,
            "values": cm[i].tolist()
        }
        for i in range(len(id2label))
    }

    cm_json_path = RUN_DIR / "confusion_matrix.json"
    with open(cm_json_path, "w", encoding="utf-8") as f:
        json.dump(cm_json, f, indent=2)
    print(f"💾 Saved confusion matrix JSON → {cm_json_path}")

    cm_png_path = RUN_DIR / "confusion_matrix.png"
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(id2label))
    plt.xticks(tick_marks, [id2label[i] for i in range(len(id2label))], rotation=45)
    plt.yticks(tick_marks, [id2label[i] for i in range(len(id2label))])

    # Annotate squares
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(cm_png_path)
    plt.close()
    print(f"Saved confusion matrix plot → {cm_png_path}")


    print("Generating learning curves...")
    history = trainer.state.log_history

    steps = []
    train_losses = []
    eval_losses = []
    eval_f1_macros = []

    for entry in history:
        if "loss" in entry and "epoch" in entry and "eval_loss" not in entry:
            # training loss logged
            train_losses.append(entry["loss"])
            steps.append(entry["epoch"])
        if "eval_loss" in entry:
            eval_losses.append(entry["eval_loss"])
        if "eval_f1_macro" in entry:
            eval_f1_macros.append(entry["eval_f1_macro"])

    # Loss curve
    plt.figure(figsize=(8, 5))
    if train_losses:
        plt.plot(train_losses, label="Train Loss")
    if eval_losses:
        plt.plot(eval_losses, label="Eval Loss")
    plt.xlabel("Logging step index")
    plt.ylabel("Loss")
    plt.title("Training & Evaluation Loss")
    plt.legend()
    plt.grid(True)
    loss_curve_path = RUN_DIR / "loss_curve.png"
    plt.tight_layout()
    plt.savefig(loss_curve_path)
    plt.close()
    print(f"Saved loss curve → {loss_curve_path}")

    # F1 curve
    if eval_f1_macros:
        plt.figure(figsize=(8, 5))
        plt.plot(eval_f1_macros, marker="o")
        plt.xlabel("Evaluation step index")
        plt.ylabel("Macro F1")
        plt.title("Eval Macro F1 over time")
        plt.grid(True)
        f1_curve_path = RUN_DIR / "f1_curve.png"
        plt.tight_layout()
        plt.savefig(f1_curve_path)
        plt.close()
        print(f"Saved F1 curve → {f1_curve_path}")


    print("Saving model and tokenizer...")
    model_dir = RUN_DIR / "model"
    model_dir.mkdir(exist_ok=True, parents=True)
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    print(f"Model & tokenizer saved to: {model_dir}")


    print("\nRunning quick prediction tests...")

    test_sentences = [
        "The hotel completely messed up my reservation, and the staff didn’t even apologize.",
        "I expected a peaceful weekend, but the room was noisy and I was disappointed.",
        "I felt happy for the first time.",
        "I was really confused during check-in, the instructions were unclear.",
        "All day I felt alone and rejected, even though my team supports me.",
        "My heart finds it s home in you, in the quiet happiness of your presence and the way you make the world feel softer.",
        "My eyes widened and my breath caught as the moment unfolded, brighter and sudded than I ever imagined."
    ]

    loaded_tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    loaded_model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

    def predict(text):
        inputs = loaded_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN
        )
        with torch.no_grad():
            logits = loaded_model(**inputs).logits
        label_id = int(torch.argmax(logits).item())
        return id2label[label_id]

    print("\nPrediction Results:")
    for t in test_sentences:
        print(f"Text: {t}")
        print(f"Predicted Emotion: {predict(t)}")
        print("-" * 50)

    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout
    print(f"📄 Full logs saved to: {log_path}")
    


if __name__ == "__main__":
    main()
