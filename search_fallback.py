import requests
from dotenv import load_dotenv
from logger import logging
import os

logger = logging.getLogger(__name__)

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")


def google_search(query, num__results=5):
    logger.info(f"Query:  {query}")
    url = 'https://www.googleapis.com/customsearch/v1'
    params = {
        'q':query,
        'key': API_KEY,
        'cx': SEARCH_ENGINE_ID,
        'num':num__results
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json().get("items", [])
        logger.info(f"Results: {results}")
        return [
            {
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet")
            }
            for item in results
        ]
    else:
        logger.error(f"Error:  {response.text}")
        raise Exception(f"Google Search failed: {response.status_code} - {response.text}")
