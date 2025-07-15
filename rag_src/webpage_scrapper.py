# Scrape text content from the specified web pages
import requests
from bs4 import BeautifulSoup


def scrape_web_pages(web_pages):
    page_contents = []
    for url in web_pages:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract all paragraph text
            paragraphs = [p.text for p in soup.find_all('p')]
            page_contents.append(" ".join(paragraphs))
        else:
            print(f"Failed to fetch {url}")
    return page_contents