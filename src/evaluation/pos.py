import pandas as pd
import spacy
from collections import Counter
import matplotlib.pyplot as plt

DATA_PATH = "../../data/emotions-dataset.xlsx"

# Load dataset into a DataFrame
df = pd.read_excel(DATA_PATH)

# Drop rows with missing 'text' or 'label' values
df = df.dropna(subset=["text", "label"])

# Load the English SpaCy model
nlp = spacy.load("en_core_web_sm")

# Function to count parts-of-speech (POS) in a text
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

# Apply POS counting to each text in the dataset
pos_data = df["text"].apply(count_pos)

# Convert the results to a DataFrame
pos_df = pd.DataFrame(list(pos_data))

# Combine the original dataset with the POS counts
df_pos = pd.concat([df.reset_index(drop=True), pos_df], axis=1)

# Group by emotion label and summarize POS counts
pos_summary = df_pos.groupby("label")[["VERB", "NOUN", "PRON", "ADJ", "ADV"]].agg(["sum", "mean", "std"])
print("POS Summary per emotion:")
print(pos_summary)

# Plot histograms for each POS count
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

# Loop through each POS type and create a histogram
for ax, pos in zip(axes, ["VERB", "NOUN", "PRON", "ADJ", "ADV", "TOTAL_TOKENS"]):
    ax.hist(df_pos[pos], bins=20, edgecolor="black")
    ax.set_title(f"{pos} count per review")
    ax.set_xlabel("Count")
    ax.set_ylabel("Frequency")

plt.tight_layout()
plt.show()

# Save POS counts per review to CSV
df_pos.to_csv("results/pos_counts_per_review.csv", index=False)
print("POS counts saved to pos_counts_per_review.csv")