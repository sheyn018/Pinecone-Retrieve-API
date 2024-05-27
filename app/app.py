import os
from dotenv import load_dotenv, find_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import pinecone

# Load environment variables
load_dotenv(find_dotenv(), override=True)
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Render assigns the PORT environment variable
port = int(os.getenv("PORT", 5000))  # Default to 5000 if not set

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the model once at startup
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize Pinecone client once at startup
pinecone.init(api_key=pinecone_api_key)
index = pinecone.Index("huggingface")

@app.route('/retrieve', methods=['GET'])
def retrieve():
    question = request.args.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400

    user_vector = model.encode(question).tolist()

    response = index.query(
        namespace="ns1",
        vector=user_vector,
        top_k=4,
        include_values=True,
        include_metadata=True
    )

    matches = response.get('matches', [])
    documents = [match['metadata']['content'] for match in matches]

    formatted_documents = ""
    for i, doc in enumerate(documents, 1):
        formatted_documents += f"Chunk Reference {i}: {doc}\n"

    return formatted_documents, 200

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=port)
