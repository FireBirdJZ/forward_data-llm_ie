import requests
from bs4 import BeautifulSoup

url = "https://cs.illinois.edu/about/people/all-faculty"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

text = soup.get_text()

word = "Professor"
positions = []
start = -1

while True:
    start = text.find(word, start + 1)
    if start == -1:
        break
    positions.append(start)

for index in positions:
    print(f"{word} found at position {index}")