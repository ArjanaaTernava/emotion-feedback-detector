import pandas as pd
import numpy as np
from datasets import Dataset

# 1. Confitguration( Konfigurimi )
file_path = "goemotions_train.csv"
df = pd.read_csv(file_path)

#  2. Label Identification and Preparation ( Identifikimi i klasave dhe pergaditja)
# Identifikimi i 28 kolonave me klasifikim ('admiration' through 'neutral')
start_col = 'admiration'
end_col = 'neutral'
start_idx = df.columns.get_loc(start_col)
end_idx = df.columns.get_loc(end_col) + 1 
emotion_columns = df.columns[start_idx:end_idx].tolist()

# Krijimi i 'labels' nje kolone me 0 dhe 1 
df['labels'] = df[emotion_columns].values.tolist()

# 3. Pergaditja e datasetit "Hugging Face"
# Konvertimi nga Pandas DataFrame ne Hugging Face Dataset
raw_dataset = Dataset.from_pandas(df[['text', 'labels']].reset_index(drop=True))

# Ndaraja e datasetit per trajnim dhe testim (90% / 10%)
# Objekti ds eshte gati per tokenizim pas ketij hapi
ds = raw_dataset.train_test_split(test_size=0.1, seed=42)