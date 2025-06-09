import os
import praw
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path

# Load credentials
load_dotenv()

#REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
#REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
#REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

# Setup Reddit API
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
    username=os.getenv("REDDIT_USERNAME"),
    
)

# Define search parameters
subreddits = ["India", "fraud", "legaladvice", "FakeIndia", "Scams", "passport"]
keywords = ["fake document", "passport scam", "Aadhaar forgery", "certificate fraud", "fake id", "forged document"]

results = []

for subreddit_name in subreddits:
    subreddit = reddit.subreddit(subreddit_name)
    print(f"🔍 Searching in r/{subreddit_name}...")

    for keyword in keywords:
        try:
            for post in subreddit.search(keyword, limit=100, sort="new"):
                results.append({
                    "id": post.id,
                    "title": post.title,
                    "text": post.selftext,
                    "subreddit": subreddit_name,
                    "author": str(post.author),
                    "created_utc": datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                    "url": post.url,
                    "keyword": keyword
                })
        except Exception as e:
            print(f"⚠️ Error while processing {subreddit_name} - {keyword}: {e}")

# Save results
Path("data/raw/reddit").mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(results)
df.to_csv("data/raw/reddit/fake_doc_reddit_posts.csv", index=False)
print(f"✅ Scraping complete. {len(results)} posts saved to 'data/raw/reddit/fake_doc_reddit_posts.csv'")
