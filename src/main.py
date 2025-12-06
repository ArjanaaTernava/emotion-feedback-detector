import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils.logging import Tee
from data.loader import load_data
from model.builder import build_model
from training.training import tokenize_dataset, train_model
from evaluation.metrics import *

# Paths and model configuration
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

    # Create directories for saving model and logs
    RUN_DIR.mkdir(exist_ok=True, parents=True)
    (RUN_DIR / "model").mkdir(exist_ok=True)

    # Redirect stdout to a log file
    sys.stdout = Tee(RUN_DIR / "logs.txt")
    print("Logging initialized")

    # Load dataset and map labels
    train_ds, test_ds, labels = load_data(DATA_PATH, RUN_DIR)

    # Build the Transformer model and tokenizer
    model, tokenizer = build_model(MODEL_NAME, len(labels))

    train_ds = tokenize_dataset(train_ds, tokenizer, CFG["MAX_LEN"])
    test_ds = tokenize_dataset(test_ds, tokenizer, CFG["MAX_LEN"])

    # Train the model
    trainer = train_model(model, train_ds, test_ds, RUN_DIR, CFG)

    # Evaluate the model and save metrics
    results = trainer.evaluate()
    save_metrics(results, RUN_DIR)

    # Make predictions on the test set
    preds = trainer.predict(test_ds)
    y_pred = preds.predictions.argmax(axis=1)
    y_true = preds.label_ids

    # Save evaluation reports and plots
    save_classification_report(y_true, y_pred, labels, RUN_DIR)
    save_confusion_matrix(y_true, y_pred, labels, RUN_DIR)
    save_learning_curves(trainer, RUN_DIR)

    # Save the trained model and tokenizer
    trainer.save_model(RUN_DIR / "model")
    tokenizer.save_pretrained(RUN_DIR / "model")

    # Load model and tokenizer for testing prediction
    id2label = {v: k for k, v in labels.items()}
    loaded_tokenizer = AutoTokenizer.from_pretrained(RUN_DIR / "model")
    loaded_model = AutoModelForSequenceClassification.from_pretrained(RUN_DIR / "model")

    test_sentences = [
        "The hotel completely messed up my reservation, and the staff didn’t even apologize.",
        "I expected a peaceful weekend, but the room was noisy and I was disappointed.",
        "I felt happy for the first time.",
        "I was really confused during check-in, the instructions were unclear.",
        "All day I felt alone and rejected, even though my team supports me.",
        "My heart finds it s home in you, in the quiet happiness of your presence and the way you make the world feel softer.",
        "My eyes widened and my breath caught as the moment unfolded, brighter and sudded than I ever imagined."
    ]

    # Quick test prediction function
    def predict(texts):
        if isinstance(texts, str):
            texts = [texts]

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CFG["MAX_LEN"]
        )

        with torch.no_grad():
            logits = model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        results = []
        for i, label_id in enumerate(preds):
            label = id2label[int(label_id)]
            confidence = probs[i][label_id].item()
            results.append({
                "text": texts[i],
                "label": label,
                "confidence": confidence
            })
        return results

    predictions = predict(test_sentences)

    for p in predictions:
        print(f"Text: {p['text']}")
        print(f"Predicted Emotion: {p['label']} ({p['confidence']:.2%} confidence)")
        print("---")


if __name__ == "__main__":
    main()
