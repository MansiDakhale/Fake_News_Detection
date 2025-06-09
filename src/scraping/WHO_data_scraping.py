# src/scraping/scrape_who.py

import requests
from bs4 import BeautifulSoup
import os
import json
from urllib.parse import urljoin

BASE_URL = "https://www.who.int"
TARGET_URL = "https://www.who.int/emergencies/diseases/novel-coronavirus-2019/advice-for-public/myth-busters"
OUTPUT_FOLDER = "data/raw/who_articles"
IMAGE_FOLDER = "images/who"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

print("Scraping WHO Mythbusters...")

response = requests.get(TARGET_URL)
soup = BeautifulSoup(response.content, 'html.parser')

articles = soup.find_all("div", class_="sf-content-block content-block")

data = []

for idx, art in enumerate(articles):
    text = art.get_text(separator=" ", strip=True)
    
    # Get image if available
    img_tag = art.find("img")
    img_path = ""
    
    if img_tag:
        img_src = img_tag.get("src")
        if img_src:
            img_url = urljoin(BASE_URL, img_src)  # handles missing scheme
            try:
                img_data = requests.get(img_url).content
                img_filename = f"who_{idx}.jpg"
                img_path = os.path.join(IMAGE_FOLDER, img_filename)
                with open(img_path, 'wb') as f:
                    f.write(img_data)
            except Exception as e:
                print(f"Could not download image: {e}")
                img_path = ""

    entry = {
        "source": "WHO",
        "title": f"WHO Myth {idx}",
        "text": text,
        "image_path": img_path,
        "is_fake": 1
    }
    data.append(entry)

# Save as JSON
with open(os.path.join(OUTPUT_FOLDER, "who_mythbusters.json"), 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print(f"✅ WHO scraping complete. Articles saved to '{OUTPUT_FOLDER}', images saved to '{IMAGE_FOLDER}'")
