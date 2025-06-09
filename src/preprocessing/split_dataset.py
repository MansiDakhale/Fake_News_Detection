# src/preprocessing/split_dataset.py

import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv(r"C:\Users\OMEN\Desktop\Fake_News_Detection\data\processed\raw_dataset.csv")

# First split: Train (70%) and temp (30%)
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)

# Second split: Validation (15%) and Test (15%) from the 30%
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

# Save the splits
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

print(f" Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")



