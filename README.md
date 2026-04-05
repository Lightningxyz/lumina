# Lumina

Lumina is a Python-based experimental text search engine that focuses on retrieval transparency. It implements the BM25 scoring algorithm alongside NLTK's Porter Stemmer to retrieve relevant documents, and provides an explainability interface to demonstrate exactly how the mathematical scoring is calculated for a given query.

Traditional search libraries often return results as a black box. Lumina instead explicitly computes and surfaces the Term Frequency (TF) and Inverse Document Frequency (IDF) contributions of each token in the query, presenting the data in a clean web interface.

## Features
- **BM25 Scoring Framework:** Uses the standard BM25 retrieval function rather than basic TF-IDF for optimal document ranking.
- **Explainable Results:** Generates a real-time explanation matrix, displaying percentage contributions and breakdown statistics for each matched term.
- **Web Interface:** A Flask-based frontend UI that renders text highlights based on their specific weight in the ranking algorithm.
- **Text Preprocessing:** Integrates NLTK stemmers via Scikit-Learn to appropriately map word variations to their root forms prior to indexing.

## Getting Started

### Prerequisites
Make sure you have Python 3 installed. Then, install the required dependencies:
```bash
pip install flask scikit-learn numpy beautifulsoup4 requests nltk
```

### 1. Generating the Dataset
To populate the search engine database for testing, you can run the dataset generator. It scrapes sections from Wikipedia to build a foundational document set:
```bash
python3 generatedataset.py
```

### 2. Running the Application
To start the Lumina web application, run the Flask backend:
```bash
python3 app.py
```

Open your browser and navigate to `http://localhost:8080/` to test the search index.

## Built With
- **Frontend:** HTML, CSS, Vanilla JS
- **Backend:** Flask
- **Engine Core:** Scikit-Learn, Numpy, NLTK
