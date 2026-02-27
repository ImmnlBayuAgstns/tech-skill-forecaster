import requests
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HNAutonomousEngine:
    def __init__(self, request_timeout: int = 10, retry_max: int = 3, rate_limit_delay: float = 0.5):
        # Algolia for "Discovery" and Firebase for "Data Extraction"
        self.discovery_api = "https://hn.algolia.com/api/v1/search_by_date"
        self.item_api = "https://hacker-news.firebaseio.com/v0/item/{}.json"

        # Configuration for API resilience
        self.request_timeout = request_timeout
        self.retry_max = retry_max
        self.rate_limit_delay = rate_limit_delay

        self.base_raw_path = Path(__file__).parent.parent.parent / "data" / "raw"

    def _request_with_retry(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Make HTTP request with retry logic and exponential backoff."""
        for attempt in range(self.retry_max):
            try:
                response = requests.get(url, params=params, timeout=self.request_timeout)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout on attempt {attempt + 1}/{self.retry_max}")
            except requests.exceptions.HTTPError as e:
                logger.warning(f"HTTP error {response.status_code} on attempt {attempt + 1}/{self.retry_max}: {e}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed on attempt {attempt + 1}/{self.retry_max}: {e}")
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse JSON response: {e}")
                return None
            
            if attempt < self.retry_max - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        
        logger.error(f"Failed to fetch {url} after {self.retry_max} attempts")
        return None

    def find_latest_thread_id(self) -> Tuple[Optional[str], Optional[str]]:
        """Dynamically finds the ID of the most recent 'Who is Hiring' thread."""
        params = {
            'query': 'Ask HN: Who is hiring?',
            'tags': 'story,author_whoishiring',
            'hitsPerPage': 1  # Only need the latest thread
        }

        # Make the API call to Algolia with retry logic
        response = self._request_with_retry(self.discovery_api, params=params)
        
        if not response or not response.get('hits'):
            logger.error("No threads found or invalid response from discovery API")
            return None, None
        
        # Extract the latest thread ID and title
        latest_hit = response['hits'][0]
        logger.info(f"Found latest thread: {latest_hit.get('title', 'Unknown')}")
        
        return str(latest_hit['objectID']), latest_hit.get('title', 'Unknown')

    def run_pipeline(self, limit: int = 200) -> None:
        """Main pipeline: Find -> Fetch -> Save"""
        # Find the latest thread ID and title
        thread_id, thread_title = self.find_latest_thread_id()
        if not thread_id:
            logger.error("Failed to find latest thread. Aborting pipeline.")
            return

        # Dynamic Filename Generation
        current_date = datetime.now().strftime("%Y-%m")
        now = datetime.now()
        partition_path = self.base_raw_path / f"year={now.year}" / f"month={now.strftime('%m')}"
        partition_path.mkdir(parents=True, exist_ok=True)
        filename = partition_path / f"{current_date}_raw_hiring.json"

        # Check if data already exists for this month
        if filename.exists():
            logger.info(f"Data for {current_date} already exists at {filename}. Skipping extraction.")
            return

        logger.info(f"Extracting: {thread_title}")

        # Extract data from the thread
        logger.info(f"Fetching data for {current_date} from thread ID {thread_id}...")
        thread_data = self._request_with_retry(self.item_api.format(thread_id))
        
        if not thread_data:
            logger.error("Failed to fetch thread data. Aborting pipeline.")
            return
        
        job_ids = thread_data.get('kids', [])[:limit]
        logger.info(f"Found {len(job_ids)} job posts to fetch")

        # Fetch each job post and filter out deleted/empty ones
        posts: List[Dict[str, Any]] = []
        for i, post_id in enumerate(job_ids):
            post = self._request_with_retry(self.item_api.format(post_id))
            
            if post and post.get('text') and not post.get('deleted'):
                posts.append({
                    'id': post.get('id'),
                    'text': post.get('text'),
                    'time': post.get('time'),
                    'company_raw': post.get('by')
                })
            
            # Rate limiting to avoid overwhelming the API
            time.sleep(self.rate_limit_delay)
            
            if (i + 1) % 50 == 0:
                logger.info(f"Progress: {i + 1}/{len(job_ids)} items fetched")

        # Save the extracted data to a JSON file
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": {
                        "id": thread_id,
                        "title": thread_title,
                        "extracted_at": datetime.now().isoformat(sep='T')
                    },
                    "posts": posts
                }, f, indent=4)
            
            logger.info(f"SUCCESS: {len(posts)} jobs saved to {filename}")
        except IOError as e:
            logger.error(f"Failed to save data to {filename}: {e}")

if __name__ == "__main__":
    try:
        engine = HNAutonomousEngine()
        engine.run_pipeline()
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error in pipeline: {e}", exc_info=True)