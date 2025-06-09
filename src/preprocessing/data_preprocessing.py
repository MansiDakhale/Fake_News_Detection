import json
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

def clean_text(text):
    """Preprocess legal case text by removing unwanted characters and stopwords."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    words = text.split()
    words = [word for word in words if word not in stopwords.words("english")]  # Remove stopwords
    return " ".join(words)

# Load scraped data
with open("indian_legal_cases.json", "r", encoding="utf-8") as f:
    cases = json.load(f)

# Clean case texts
for case in cases:
    case["judgment_text"] = clean_text(case["judgment_text"])

# Save cleaned data
with open("cleaned_legal_cases.json", "w", encoding="utf-8") as f:
    json.dump(cases, f, indent=4)

print(" Data cleaned and saved as 'cleaned_legal_cases.json'")
