import pandas as pd
import spacy
from collections import Counter
import matplotlib.pyplot as plt

DATA_PATH = "../../data/emotions-dataset.xlsx"
df = pd.read_excel(DATA_PATH)
df = df.dropna(subset=["text", "label"])

nlp = spacy.load("en_core_web_sm")


def count_pos(text):
    doc = nlp(text)
    pos_counts = Counter(token.pos_ for token in doc)
    return {
        "VERB": pos_counts.get("VERB", 0),
        "NOUN": pos_counts.get("NOUN", 0),
        "PRON": pos_counts.get("PRON", 0),
        "ADJ": pos_counts.get("ADJ", 0),
        "ADV": pos_counts.get("ADV", 0),
        "TOTAL_TOKENS": len(doc)
    }

pos_data = df["text"].apply(count_pos)
pos_df = pd.DataFrame(list(pos_data))

df_pos = pd.concat([df.reset_index(drop=True), pos_df], axis=1)


pos_summary = df_pos.groupby("label")[["VERB", "NOUN", "PRON", "ADJ", "ADV"]].agg(["sum", "mean", "std"])
print("POS Summary per emotion:")
print(pos_summary)


plt.figure(figsize=(12, 5))
for i, pos in enumerate(["VERB", "NOUN", "PRON", "ADJ", "ADV", "TOTAL_TOKENS"]):
    plt.subplot(1, 3, i+1)
    plt.hist(df_pos[pos], bins=20, color="skyblue", edgecolor="black")
    plt.title(f"{pos} count per review")
    plt.xlabel("Count")
    plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


df_pos.to_csv("results/pos_counts_per_review.csv", index=False)
print("POS counts saved to pos_counts_per_review.csv")