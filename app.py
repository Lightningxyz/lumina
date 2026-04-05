from flask import Flask, request, jsonify, render_template
from engine import BM25Engine, load_documents

app = Flask(__name__)

# Initialize search engine at startup
print("Starting search engine...")
docs = load_documents("documents.txt")
engine = BM25Engine()
engine.fit(docs)
print(f"Engine fitted with {len(docs)} documents.")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query", "")
    top_k = data.get("top_k", 10)
    
    if not query:
        return jsonify({"results": []})
        
    results = engine.search(query, top_k=top_k)
    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(debug=True, port=8080)
