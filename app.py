import os
from dotenv import load_dotenv, find_dotenv
from flask import Flask, request
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Load environment variables
load_dotenv(find_dotenv(), override=True)
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize Flask app and CORS
app = Flask(__name__)
CORS(app)

@app.route('/retrieve', methods=['GET'])
def retrieve():
    # Get the question from the request
    question = request.args.get('question')
    print("Got question: ", question)

    # Encode the question into a vector using SentenceTransformer
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    user_vector = model.encode(question).tolist()

    # Initialize Pinecone and query the index
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index("huggingface")

    response = index.query(
        namespace="ns1",
        vector=user_vector,
        top_k=4,
        include_values=True,
        include_metadata=True
    )

    # Extract and format the matched documents
    matches = response['matches']
    documents = [match['metadata']['content'] for match in matches]

    formatted_documents = "\n".join([f"Chunk Reference {i}: {doc}" for i, doc in enumerate(documents, 1)])

    return formatted_documents, 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
