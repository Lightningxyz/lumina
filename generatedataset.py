import requests
from bs4 import BeautifulSoup

TOPICS = [
    "Machine_learning",
    "Artificial_intelligence",
    "Neural_network",
    "Deep_learning",
    "Natural_language_processing",
    "Computer_vision"
]

def fetch_wikipedia_text(topic):
    url = f"https://en.wikipedia.org/wiki/{topic}"
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.text, "html.parser")

    paragraphs = soup.find_all("p")
    text = []

    for p in paragraphs:
        content = p.get_text().strip()

        if len(content) > 100:
            text.append(content)

    return text

def clean_and_split(paragraphs):
    sentences = []

    for para in paragraphs:
        parts = para.split(".")
        for part in parts:
            line = part.strip()

            if 50 < len(line) < 200:
                sentences.append(line)

    return sentences

def build_dataset():
    all_sentences = []

    for topic in TOPICS:
        print(f"Fetching {topic}...")
        paragraphs = fetch_wikipedia_text(topic)
        sentences = clean_and_split(paragraphs)
        all_sentences.extend(sentences)

    with open("documents.txt", "w") as f:
        for line in all_sentences:
            f.write(line + "\n")

    print(f"\nDataset created with {len(all_sentences)} documents.")

if __name__ == "__main__":
    build_dataset()