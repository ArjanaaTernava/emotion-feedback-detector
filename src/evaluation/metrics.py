import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


def save_metrics(results, run_dir):
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)


def save_classification_report(y_true, y_pred, labels, run_dir):

    id2label = {v: k for k, v in labels.items()}

    report = classification_report(
        y_true,
        y_pred,
        target_names=[id2label[i] for i in range(len(id2label))],
        digits=4
    )

    print(report)

    with open(run_dir / "classification_report.txt", "w") as f:
        f.write(report)


def save_confusion_matrix(y_true, y_pred, labels, run_dir):

    cm = confusion_matrix(y_true, y_pred)
    id2label = {v: k for k, v in labels.items()}

    cm_json = {
        id2label[i]: {"row": i, "values": cm[i].tolist()}
        for i in range(len(id2label))
    }

    with open(run_dir / "confusion_matrix.json", "w") as f:
        json.dump(cm_json, f, indent=2)

    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(id2label))
    plt.xticks(ticks, id2label.values(), rotation=45)
    plt.yticks(ticks, id2label.values())

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    plt.savefig(run_dir / "confusion_matrix.png")
    plt.close()


def save_learning_curves(trainer, run_dir):

    history = trainer.state.log_history
    train_loss, eval_loss, f1 = [], [], []

    for entry in history:
        if "loss" in entry and "eval_loss" not in entry:
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_loss.append(entry["eval_loss"])
        if "eval_f1_macro" in entry:
            f1.append(entry["eval_f1_macro"])

    plt.plot(train_loss, label="Train")
    plt.plot(eval_loss, label="Eval")
    plt.legend()
    plt.title("Loss")
    plt.savefig(run_dir / "loss_curve.png")
    plt.close()

    plt.plot(f1, marker='o')
    plt.title("F1 Macro")
    plt.savefig(run_dir / "f1_curve.png")
    plt.close()
