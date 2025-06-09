import json
import pandas as pd

# Load your WHO JSON
with open(r"C:\Users\OMEN\Desktop\Fake_News_Detection\data\raw\who_articles\who_mythbusters.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Optional: Clean up title and text fields
df["text"] = df["title"] + ". " + df["text"]
df["label"] = df["is_fake"].apply(lambda x: "FAKE" if x == 1 else "REAL")
df["modality"] = df["image_path"].apply(lambda x: "image-text" if x else "text-only")
df["image_url"] = df["image_path"]
df["source"] = "WHO"

# Select final columns
final_df = df[["text", "label", "source", "image_url", "modality"]]

# Save to CSV
final_df.to_csv("data/processed/who_data.csv", index=False)
print("✅ WHO data saved as CSV.")
