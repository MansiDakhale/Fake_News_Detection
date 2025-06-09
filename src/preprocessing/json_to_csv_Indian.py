import json
import pandas as pd

# Load your JSON file
with open(r"C:\Users\OMEN\Desktop\Fake_News_Detection\data\raw\indian_kanoon\cleaned_legal_cases.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Drop or replace meaningless data
df = df.replace("Unknown Title", "")
df = df.replace("Unknown Date", "")
df = df.replace("Unknown Judges", "")
df = df[df["judgment_text"].str.len() > 30]  # Remove very short entries

# Add required fields for deep learning pipeline
df["text"] = df["title"] + ". " + df["judgment_text"]
df["label"] = "FAKE"  # or "REAL" — based on your pipeline
df["source"] = "Indian Kanoon"
df["image_url"] = ""
df["modality"] = "text-only"

# Keep only the final columns
final_df = df[["text", "label", "source", "image_url", "modality", "date", "url"]]

# Save to CSV
final_df.to_csv("data/processed/indian_kanoon.csv", index=False)
print("✅ Indian Kanoon data saved as CSV.")
