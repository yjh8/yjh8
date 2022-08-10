import json
import csv
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to read bookmarks from a JSON or CSV file
def read_bookmarks(file_path):
    bookmarks = []
    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            bookmarks = json.load(f)
    elif file_path.endswith('.csv'):
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            bookmarks = [row for row in reader]
    return bookmarks

# Function to extract keywords using spaCy
def extract_keywords(text):
    doc = nlp(text)
    keywords = [token.text for token in doc if token.is_alpha and not token.is_stop]
    return keywords

# Function to classify bookmarks into categories using KMeans
def classify_bookmarks(bookmarks, num_categories=5):
    descriptions = [bookmark['description'] for bookmark in bookmarks]
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(descriptions)
    kmeans = KMeans(n_clusters=num_categories, random_state=42)
    labels = kmeans.fit_predict(X)
    for i, bookmark in enumerate(bookmarks):
        bookmark['category'] = f"Category {labels[i]}"
    return bookmarks

# Function to generate tags based on content
def generate_tags(bookmark):
    text = bookmark['description']
    keywords = extract_keywords(text)
    stop_words = set(stopwords.words('english'))
    filtered_keywords = [word for word in keywords if word.lower() not in stop_words]
    return list(set(filtered_keywords))

# Function to save organized bookmarks back to a structured file
def save_bookmarks(bookmarks, output_file):
    if output_file.endswith('.json'):
        with open(output_file, 'w') as f:
            json.dump(bookmarks, f, indent=4)
    elif output_file.endswith('.csv'):
        with open(output_file, 'w', newline='') as f:
            fieldnames = bookmarks[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(bookmarks)

# Main function
def main(input_file, output_file):
    bookmarks = read_bookmarks(input_file)
    bookmarks = classify_bookmarks(bookmarks)
    for bookmark in bookmarks:
        bookmark['tags'] = generate_tags(bookmark)
    save_bookmarks(bookmarks, output_file)

if __name__ == "__main__":
    input_file = "bookmarks.json"  # Replace with your input file path
    output_file = "organized_bookmarks.json"  # Replace with your output file path
    main(input_file, output_file)