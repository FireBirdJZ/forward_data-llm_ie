import requests

# Send a GET request to the URL
url = "https://cs.illinois.edu/about/people/all-faculty"
response = requests.get(url)

# Get the text content of the response
text = response.text

# Find the position of the word 'Professor' in the text
start_pos = 0
while True:
    pos = text.find("Professor", start_pos)
    if pos == -1:
        break
    print(f"Found 'Professor' at position {pos}")
    start_pos = pos + 1