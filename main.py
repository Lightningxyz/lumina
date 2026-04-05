import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

TOP_K = 3

def load_documents(file_path):
    with open(file_path, 'r') as f:
        docs = [line.strip() for line in f.readlines() if line.strip()]
    return docs

def build_tfidf(docs):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(docs)

    feature_names = vectorizer.get_feature_names_out()
    idf_values = dict(zip(feature_names, vectorizer.idf_))

    return vectorizer, tfidf_matrix, feature_names, idf_values

def compute_tf(doc, feature_names):
    words = doc.lower().split()
    tf_dict = {}

    for word in words:
        if word in feature_names:
            tf_dict[word] = tf_dict.get(word, 0) + 1

    total = len(words)
    for word in tf_dict:
        tf_dict[word] /= total

    return tf_dict

def search(query, vectorizer, tfidf_matrix, docs, feature_names, idf_values):
    query_vec = vectorizer.transform([query])
    scores = np.dot(tfidf_matrix, query_vec.T).toarray().flatten()

    ranked_indices = np.argsort(scores)[::-1][:TOP_K]

    results = []

    query_vector = query_vec.toarray().flatten()

    for idx in ranked_indices:
        if scores[idx] <= 0:
            continue

        doc_vector = tfidf_matrix[idx].toarray().flatten()
        doc = docs[idx]

        explanation = []
        total_contribution = 0

        tf_values = compute_tf(doc, feature_names)

        for i in range(len(feature_names)):
            if query_vector[i] > 0 and doc_vector[i] > 0:
                term = feature_names[i]
                contribution = doc_vector[i] * query_vector[i]
                total_contribution += contribution

                explanation.append({
                    "term": term,
                    "contribution": contribution,
                    "tf": tf_values.get(term, 0),
                    "idf": idf_values.get(term, 0),
                    "tfidf": doc_vector[i]
                })

        # Normalize contributions to %
        for item in explanation:
            item["percent"] = (item["contribution"] / total_contribution) * 100

        explanation.sort(key=lambda x: x["contribution"], reverse=True)

        results.append({
            "doc": doc,
            "score": scores[idx],
            "explanation": explanation
        })

    return results

def print_results(results):
    for i, res in enumerate(results):
        print(f"\nResult {i+1}")
        print("Document:", res["doc"])
        print("Score:", round(res["score"], 4))
        print("Explanation (Detailed):")

        for item in res["explanation"]:
            print(f"""
  Term: {item["term"]}
    Contribution: {round(item["contribution"], 4)}
    Percent: {round(item["percent"], 2)}%
    TF: {round(item["tf"], 4)}
    IDF: {round(item["idf"], 4)}
    TF-IDF: {round(item["tfidf"], 4)}
""")
        print("-" * 50)

def main():
    docs = load_documents("documents.txt")

    vectorizer, tfidf_matrix, feature_names, idf_values = build_tfidf(docs)

    print("Explainable TF-IDF Search Engine")
    print("--------------------------------")

    while True:
        query = input("\nEnter query (or 'exit'): ")

        if query.lower() == "exit":
            break

        results = search(query, vectorizer, tfidf_matrix, docs, feature_names, idf_values)

        if not results:
            print("No relevant documents found.")
        else:
            print_results(results)

if __name__ == "__main__":
    main()