import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments


def tokenize_dataset(dataset, tokenizer, max_len):

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_len
        )

    dataset = dataset.map(tokenize, batched=True)
    keep = ["input_ids", "attention_mask", "label"]

    dataset = dataset.remove_columns(
        [c for c in dataset.column_names if c not in keep]
    )
    dataset.set_format("torch")

    return dataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def train_model(model, train_ds, test_ds, run_dir, cfg):

    args = TrainingArguments(
        output_dir=str(run_dir / "model"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=cfg["TRAIN_BATCH"],
        per_device_eval_batch_size=cfg["EVAL_BATCH"],
        num_train_epochs=cfg["EPOCHS"],
        learning_rate=cfg["LR"],
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.train()

    return trainer
