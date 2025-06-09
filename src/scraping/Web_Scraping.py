import requests
from bs4 import BeautifulSoup
import time
import json
import random

# Keywords to collect diverse cases
LEGAL_KEYWORDS = [
    "fraud", "corruption", "contract dispute", "bribery", 
    "misrepresentation", "cyber crime", "defamation", "forgery", "cheating"
]

NUM_PAGES = 200  # Increase to collect more data
OUTPUT_FILE = "large_legal_cases.json"


def get_case_links(query, num_pages=100):
    """Scrape case links from Indian Kanoon search results."""
    base_url = "https://www.indiankanoon.org/search/?formInput="
    case_links = set()

    for page in range(num_pages):
        url = f"{base_url}{query}&p={page}"
        headers = {"User-Agent": "Mozilla/5.0"}  # Imitate a browser request
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print(f" Failed to fetch page {page} for '{query}'. Skipping...")
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract case links
        for link in soup.select("a[href^='/doc/']"):
            case_links.add("https://www.indiankanoon.org" + link["href"])

        print(f"Scraped {len(case_links)} links for '{query}', Page {page}")
        
        time.sleep(random.uniform(2, 5))  # Random delay to prevent blocking

    return list(case_links)


def scrape_case(url):
    """Extract detailed information from a case page."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f" Failed to fetch case {url}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract case details
    title = soup.find("h1").text.strip() if soup.find("h1") else "Unknown Title"
    date = soup.find("meta", {"name": "DC.date"})["content"] if soup.find("meta", {"name": "DC.date"}) else "Unknown Date"
    judges = soup.find("div", class_="judges").text.strip() if soup.find("div", class_="judges") else "Unknown Judges"

    # Extract full case text
    paragraphs = soup.find_all("p")
    judgment_text = "\n".join([p.text for p in paragraphs])[:10000]  # Store up to 10,000 chars

    return {
        "title": title,
        "date": date,
        "judges": judges,
        "url": url,
        "judgment_text": judgment_text
    }


# Start Scraping Process
all_case_links = set()

for keyword in LEGAL_KEYWORDS:
    case_links = get_case_links(keyword, num_pages=NUM_PAGES)
    all_case_links.update(case_links)

print(f"\n Total unique case links collected: {len(all_case_links)}")

# Scrape detailed case data
cases_data = [scrape_case(link) for link in all_case_links if link]

# Save to JSON file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(cases_data, f, indent=4)

print(f"\nScraped {len(cases_data)} cases and saved to '{OUTPUT_FILE}'")


