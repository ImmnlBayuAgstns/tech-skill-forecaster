import requests
import json
import time
from datetime import datetime
from pathlib import Path

class HNAutonomousEngine:
    def __init__(self):
        # Algolia for "Discovery" and Firebase for "Data Extraction"
        self.discovery_api = "https://hn.algolia.com/api/v1/search_by_date"
        self.item_api = "https://hacker-news.firebaseio.com/v0/item/{}.json"

        # Set up raw data directory
        self.raw_data_path = Path(__file__).parent.parent.parent / "data" / "raw"
        self.raw_data_path.mkdir(parents=True, exist_ok=True)

    def find_latest_thread_id(self):
        """Dynamically finds the ID of the most recent 'Who is Hiring' thread."""
        params = {
            'query': 'Ask HN: Who is hiring?',
            'tags': 'story,author_whoishiring',
            'hitsPerPage': 1 # Only need the latest thread
        }

        # Make the API call to Algolia
        try:
            response = requests.get(self.discovery_api, params=params).json()
            if not response['hits']:
                return None, None
                
            # Extract the latest thread ID and title
            latest_hit = response['hits'][0]
            print(f"📡 Found Latest Thread: {latest_hit['title']}")
            
            return str(latest_hit['objectID']), latest_hit['title']
        
        except Exception as e:
            print(f"❌ Error during discovery: {e}")
            return None, None

    def run_pipeline(self, limit=200):
        """Main pipeline: Find -> Fetch -> Save"""
        # Find the latest thread ID and title
        thread_id, thread_title = self.find_latest_thread_id()
        if not thread_id: return

        # Dynamic Filename Generation
        current_date = datetime.now().strftime("%Y-%m")
        filename = self.raw_data_path / f"{current_date}_raw.json"

        print(f"📥 Extracting: {thread_title}")

        # Extract data from the thread
        print(f"📥 Fetching data for {current_date} from ID {thread_id}...")
        thread_data = requests.get(self.item_api.format(thread_id)).json()
        job_ids = thread_data.get('kids', [])[:limit]

        # Fetch each job post and filter out deleted/empty ones
        posts = []
        for i, post_id in enumerate(job_ids):
            post = requests.get(self.item_api.format(post_id)).json()
            if post and post.get('text') and not post.get('deleted'):
                posts.append({
                    'id': post.get('id'),
                    'text': post.get('text'),
                    'time': post.get('time'),
                    'company_raw': post.get('by')
                })
            if (i + 1) % 50 == 0:
                print(f"Progress: {i + 1}/{len(job_ids)} items...")

        # Save the extracted data to a JSON file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {"id": thread_id, "title": thread_title, "extracted_at": datetime.now().isoformat(sep='T')},
                "posts": posts
            }, f, indent=4)
        
        print(f"✅ SUCCESS: {len(posts)} jobs saved to {filename}")

if __name__ == "__main__":
    engine = HNAutonomousEngine()
    engine.run_pipeline()