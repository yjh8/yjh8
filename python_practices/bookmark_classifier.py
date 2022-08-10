import json
import os
import re
import nltk
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

nltk.download("stopwords")
nltk.download("punkt")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load bookmarks from JSON file
def load_bookmarks(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Extract text from a URL
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        return " ".join([p.text for p in soup.find_all("p")])
    except:
        return ""

# Preprocess text
def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text.lower())
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    return " ".join(words)

# Classify bookmarks using KMeans clustering
def classify_bookmarks(bookmarks):
    texts = [extract_text_from_url(b["url"]) for b in bookmarks]
    processed_texts = [preprocess_text(text) for text in texts]
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed_texts)
    
    num_clusters = 5  # Adjust based on your dataset
    model = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = model.fit_predict(X)
    
    for i, bookmark in enumerate(bookmarks):
        bookmark["category"] = f"Cluster {clusters[i]}"
    
    return bookmarks

# Tag bookmarks based on frequent words
def generate_tags(bookmarks):
    for bookmark in bookmarks:
        words = bookmark.get("title", "") + " " + bookmark.get("category", "")
        words = preprocess_text(words).split()
        common_words = [word for word, _ in Counter(words).most_common(3)]
        bookmark["tags"] = common_words
    return bookmarks

# Save updated bookmarks
def save_bookmarks(bookmarks, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(bookmarks, f, indent=4)

if __name__ == "__main__":
    input_file = "bookmarks.json"
    output_file = "classified_bookmarks.json"
    
    bookmarks = load_bookmarks(input_file)
    bookmarks = classify_bookmarks(bookmarks)
    bookmarks = generate_tags(bookmarks)
    save_bookmarks(bookmarks, output_file)
    
    print(f"Bookmarks classified and saved to {output_file}")
