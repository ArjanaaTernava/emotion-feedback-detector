import json
import pandas as pd
from pathlib import Path
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import f1_score, accuracy_score, classification_report

CSV_PATH = "data/reviews_with_custom_emotions.xlsx"
TEXT_COLUMN = "text"

FIRST_EMOTION = "admiration"
LAST_EMOTION = "neutral"

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = Path("emotion_model")
MAX_LEN = 128
BATCH = 16
EPOCHS = 3
LR = 2e-5

class MultiLabelCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)

        labels = []
        for f in features:
            l = f["labels"]

            if isinstance(l, torch.Tensor):
                l = l.tolist()

            if isinstance(l[0], (list, tuple)):
                flat = []
                for sub in l:
                    if isinstance(sub, (list, tuple)):
                        flat.extend(sub)
                    else:
                        flat.append(sub)
                l = flat

            l = [float(x) for x in l]
            labels.append(l)

        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        return batch


def load_dataset():
    df = pd.read_csv(CSV_PATH)

    start = df.columns.get_loc(FIRST_EMOTION)
    end = df.columns.get_loc(LAST_EMOTION) + 1
    emotion_cols = df.columns[start:end].tolist()

    with open("labels.json", "w") as f:
        json.dump(emotion_cols, f, indent=2)

    df["labels"] = df[emotion_cols].values.tolist()

    ds = Dataset.from_pandas(df[[TEXT_COLUMN, "labels"]])
    ds = ds.train_test_split(test_size=0.1, seed=42)

    return DatasetDict({"train": ds["train"], "test": ds["test"]})

def tokenize(ds, tok):
    def process(batch):
        return tok(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )
    return ds.map(process, batched=True)

def metrics_fn(pred):
    logits, labels = pred
    preds = (logits > 0).astype(int)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }

def train():
    print("📥 Loading dataset...")
    ds = load_dataset()

    print("🔑 Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("✂️ Tokenizing...")
    ds = tokenize(ds, tok)

    print("🔧 Normalizing labels...")

    def normalize(batch):
        out = []
        for lbl in batch["labels"]:
            if isinstance(lbl, torch.Tensor):
                lbl = lbl.tolist()
            out.append([float(x) for x in lbl])
        return {"labels": out}

    ds = ds.map(normalize, batched=True)

    print("🧠 Loading model...")

    num_labels = len(ds["train"][0]["labels"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        learning_rate=LR,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        num_train_epochs=EPOCHS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        report_to="none",
    )

    collator = MultiLabelCollator(tokenizer=tok)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=metrics_fn,
    )

    print("🚀 Training...")
    trainer.train()

    print("💾 Saving model...")
    trainer.save_model()
    tok.save_pretrained(OUTPUT_DIR)

    print("📊 Final evaluation:")
    preds = trainer.predict(ds["test"])
    preds_bin = (preds.predictions > 0).astype(int)
    labels = preds.label_ids

    print(classification_report(labels, preds_bin, zero_division=0))

if __name__ == "__main__":
    train()
