# Lumina

**Lumina** is a lightweight, high-performance "glass box" search engine. Unlike standard black-box search algorithms that simply return a list of documents without context, Lumina explicitly quantifies and highlights *why* a particular document matches your query. 

It accomplishes this by using an optimized **BM25 algorithm** paired with **NLTK Porter Stemming**. When you execute a search, Lumina calculates the term frequency and inverse document frequency of every relevant token, and dynamically renders the textual resonance of those variables on a beautiful, glassmorphic frontend UI.

## Features
- 🔍 **BM25 Scoring Framework:** Uses the industry-standard BM25 retrieval function rather than basic TF-IDF for state-of-the-art accuracy.
- 🪟 **Glass Box Explainability:** Generates a real-time "Neural Explanation Matrix", assigning percentage contributions and breakdown statistics to every highlighted word in a document.
- 🎨 **Neumorphic / Cyber Aesthetic:** A stunning, modern Flask UI utilizing deep mesh gradients, translucent cards, and dynamic HTML injection.
- 🧠 **Advanced Natural Language Processing:** Implements an NLTK Stemmer via Scikit-Learn to map varying word suffixes to their root form (e.g., treating "computation" and "computing" identically).

## Getting Started

### Prerequisites
Make sure you have Python 3 installed. Then, install the required dependencies:
```bash
pip install flask scikit-learn numpy beautifulsoup4 requests nltk
```

### 1. Generating the Dataset
To populate the search engine database, simply run the dataset generator. It scrapes relevant domains from Wikipedia to build a structured foundational document:
```bash
python3 generatedataset.py
```

### 2. Booting the Core
To start the Lumina web application, run the Flask backend:
```bash
python3 app.py
```
*Note: During startup, Lumina will tokenize the entire dataset and precompute the Inverse Document Frequencies across the matrix. A dataset of 1,200 paragraphs takes roughly ~0.5 seconds to mount.*

Open your browser and navigate to `http://localhost:8080/` to begin executing searches.

## Built With
- **Frontend:** HTML, CSS Variables, Vanilla JS
- **Backend:** Flask (Python)
- **Engine Core:** Scikit-Learn Vectors, Numpy Math Matrices, NLTK
