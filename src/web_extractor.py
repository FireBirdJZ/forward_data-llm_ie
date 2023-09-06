import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import re

def ExtractTextFromWebpage(url):
    """
    Extract all text content from a webpage.

    Args:
        url (str): The URL of the webpage to extract text from.

    Returns:
        str: The extracted text from the webpage with extra whitespace removed.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract all the text from the webpage
        page_text = soup.get_text()

        # Remove extra whitespace using regular expressions
        cleaned_text = re.sub(r'\s+', ' ', page_text)

        return cleaned_text.strip()  # Remove leading/trailing whitespace
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    # Example usage
    webpage_url = "https://cs.illinois.edu/about/people/faculty/jeffe"
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(ExtractTextFromWebpage, webpage_url)
        result = future.result()

    if result:
        print(result)
    else:
        print("Failed to extract text from the webpage.")
