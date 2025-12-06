import json
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

EMOTION_LABELS = {
    "sadness": 0,
    "joy": 1,
    "love": 2,
    "anger": 3,
    "fear": 4,
    "surprise": 5,
}


def load_data(data_path, run_dir):

    print("Loading dataset...")
    df = pd.read_excel(data_path)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Excel file must contain 'text' and 'label' columns.")

    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].map(EMOTION_LABELS)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    with open(run_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(EMOTION_LABELS, f, indent=2)
    print("Saved label mapping")

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )

    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    return (
        Dataset.from_pandas(train_df.reset_index(drop=True)),
        Dataset.from_pandas(test_df.reset_index(drop=True)),
        EMOTION_LABELS
    )
