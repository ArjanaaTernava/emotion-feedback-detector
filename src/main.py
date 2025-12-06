import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils.logging import Tee
from data.loader import load_data
from model.builder import build_model
from training.training import tokenize_dataset, train_model
from evaluation.metrics import *

DATA_PATH = "../data/emotions-dataset.xlsx"
RUN_DIR = Path("preprocess-3-epochs-bert-model")
MODEL_NAME = "distilbert-base-uncased"

CFG = {
    "MAX_LEN": 128,
    "EPOCHS": 3,
    "LR": 2e-5,
    "TRAIN_BATCH": 16,
    "EVAL_BATCH": 32,
}


def main():

    RUN_DIR.mkdir(exist_ok=True, parents=True)
    (RUN_DIR / "model").mkdir(exist_ok=True)

    sys.stdout = Tee(RUN_DIR / "logs.txt")
    print("Logging initialized")

    train_ds, test_ds, labels = load_data(DATA_PATH, RUN_DIR)

    model, tokenizer = build_model(MODEL_NAME, len(labels))

    train_ds = tokenize_dataset(train_ds, tokenizer, CFG["MAX_LEN"])
    test_ds = tokenize_dataset(test_ds, tokenizer, CFG["MAX_LEN"])

    trainer = train_model(model, train_ds, test_ds, RUN_DIR, CFG)

    results = trainer.evaluate()
    save_metrics(results, RUN_DIR)

    preds = trainer.predict(test_ds)
    y_pred = preds.predictions.argmax(axis=1)
    y_true = preds.label_ids

    save_classification_report(y_true, y_pred, labels, RUN_DIR)
    save_confusion_matrix(y_true, y_pred, labels, RUN_DIR)
    save_learning_curves(trainer, RUN_DIR)

    trainer.save_model(RUN_DIR / "model")
    tokenizer.save_pretrained(RUN_DIR / "model")

    id2label = {v: k for k, v in labels.items()}

    loaded_tokenizer = AutoTokenizer.from_pretrained(RUN_DIR / "model")
    loaded_model = AutoModelForSequenceClassification.from_pretrained(RUN_DIR / "model")

    def predict(text):
        inputs = loaded_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CFG["MAX_LEN"]
        )
        with torch.no_grad():
            logits = loaded_model(**inputs).logits
        return id2label[int(torch.argmax(logits))]

    print("Prediction test:")
    print(predict("I felt extremely happy today!"))


if __name__ == "__main__":
    main()
