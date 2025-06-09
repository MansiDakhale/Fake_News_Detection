import pandas as pd
import os
import requests
from PIL import Image
from io import BytesIO

# Load your dataset
df = pd.read_csv(r"C:\Users\OMEN\Desktop\Fake_News_Detection\data\splitted\train.csv")

# Directory to store images
image_dir = "Train_images"
os.makedirs(image_dir, exist_ok=True)

# Download images
def download_image(row):
    image_url = row["image_url"]
    image_id = row["id"]
    try:
        response = requests.get(image_url, timeout=5)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content)).convert("RGB")
            path = os.path.join(image_dir, f"{image_id}.jpg")
            image.save(path)
            return f"{image_id}.jpg"
        else:
            return None
    except:
        return None

df["local_image_path"] = df.apply(download_image, axis=1)

# Remove rows where image failed to download
df = df[df["local_image_path"].notnull()]

# Save cleaned and updated CSV
df[["text", "local_image_path", "label"]].to_csv("train.csv", index=False)