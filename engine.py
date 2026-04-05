import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import math
from nltk.stem.porter import PorterStemmer

class BM25Engine:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.vectorizer = None
        self.feature_names = []
        self.idf_values = {}
        self.doc_vectors = None
        self.docs = []
        self.avgdl = 0
        self.doc_lengths = []
        self.stemmer = PorterStemmer()

    def build_analyzer(self):
        cv = CountVectorizer(stop_words='english')
        analyzer = cv.build_analyzer()
        return lambda doc: [self.stemmer.stem(w) for w in analyzer(doc)]

    def fit(self, docs):
        self.docs = docs
        self.vectorizer = CountVectorizer(analyzer=self.build_analyzer())
        
        # Sparse matrix of term frequencies (TF)
        self.doc_vectors = self.vectorizer.fit_transform(docs)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        num_docs = self.doc_vectors.shape[0]
        # Precompute document lengths and average document length
        self.doc_lengths = self.doc_vectors.sum(axis=1).A1
        self.avgdl = self.doc_lengths.mean()
        
        # Compute IDF for all terms
        # Document frequency (DF): number of documents containing term i
        df = np.bincount(self.doc_vectors.indices, minlength=self.doc_vectors.shape[1])
        
        # Standard BM25 IDF formula
        self.idf_values = np.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
        
    def search(self, query, top_k=5):
        analyzer = self.build_analyzer()
        query_terms = analyzer(query)
        
        # Get query token indices
        query_term_indices = []
        for term in set(query_terms):
            if term in self.vectorizer.vocabulary_:
                query_term_indices.append(self.vectorizer.vocabulary_[term])
        
        if not query_term_indices:
            return []
            
        scores = np.zeros(len(self.docs))
        doc_contributions = [[] for _ in range(len(self.docs))]
        
        for idx in query_term_indices:
            feature_name = self.feature_names[idx]
            idf = self.idf_values[idx]
            
            tf_column = self.doc_vectors[:, idx].toarray().flatten()
            non_zero_docs = np.nonzero(tf_column)[0]
            
            for doc_id in non_zero_docs:
                tf = tf_column[doc_id]
                dl = self.doc_lengths[doc_id]
                
                # BM25 term weighting
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (dl / self.avgdl))
                
                contribution = idf * (numerator / denominator)
                scores[doc_id] += contribution
                
                doc_contributions[doc_id].append({
                    "term": feature_name,
                    "contribution": float(contribution),
                    "tf": int(tf),
                    "idf": float(idf)
                })

        # Rank documents
        ranked_indices = np.argsort(scores)[::-1]
        results = []
        
        for doc_id in ranked_indices[:top_k]:
            if scores[doc_id] <= 0:
                continue
                
            explanation = doc_contributions[doc_id]
            total_contribution = scores[doc_id]
            
            # Add percentage
            for item in explanation:
                item["percent"] = (item["contribution"] / total_contribution) * 100
                
            explanation.sort(key=lambda x: x["contribution"], reverse=True)
            
            results.append({
                "doc_id": int(doc_id),
                "doc": self.docs[doc_id],
                "score": float(scores[doc_id]),
                "explanation": explanation
            })
            
        return results

def load_documents(file_path):
    with open(file_path, 'r') as f:
        docs = [line.strip() for line in f.readlines() if line.strip()]
    return docs
