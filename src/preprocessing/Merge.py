import pandas as pd
import uuid

# Load datasets
fakeddit_path = r"C:\Users\OMEN\Desktop\Fake_News_Detection\data\processed\Multi-modal Dataset.csv"
boomlive_path = r"C:\Users\OMEN\Desktop\Fake_News_Detection\data\processed\news_dataset_cleaned.csv"

# Fakeddit Dataset
df_fakeddit = pd.read_csv(fakeddit_path, dtype={"id": str}, low_memory=False)

# Ensure 'image_url' column exists and rename if needed
if 'image_url' not in df_fakeddit.columns and 'url' in df_fakeddit.columns:
    df_fakeddit = df_fakeddit.rename(columns={"url": "image_url"})

df_fakeddit["label"] = df_fakeddit["label"].astype(int)
df_fakeddit["url"] = None  # Fakeddit doesn't have a webpage URL
df_fakeddit = df_fakeddit[["id", "text", "url", "image_url", "label"]]

# Boomlive Dataset
df_boom = pd.read_csv(boomlive_path)

# Generate UUIDs as IDs
df_boom["id"] = [str(uuid.uuid4()) for _ in range(len(df_boom))]

# Rename columns to match target schema
df_boom = df_boom.rename(columns={
    "text": "text",
    "image": "image_url",
    "url": "url",
    "label": "label"
})

# Normalize labels
df_boom["label"] = df_boom["label"].map({"fake": 1, "real": 0})
df_boom = df_boom[["id", "text", "url", "image_url", "label"]]

# Combine both
df_combined = pd.concat([df_fakeddit, df_boom], ignore_index=True)

# Drop rows with missing text or image_url
df_combined.dropna(subset=["text", "image_url", "label"], inplace=True)

# Save to file
output_path = r"C:\Users\OMEN\Desktop\Fake_News_Detection\data\processed\raw_dataset.csv"
df_combined.to_csv(output_path, index=False)

print(f"✅ Cleaned and merged dataset saved to: {output_path}")
print(df_combined.head())
