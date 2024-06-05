import os
from dotenv import load_dotenv, find_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Load environment variables
load_dotenv(find_dotenv(), override=True)
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Initialize Flask app and CORS
app = Flask(__name__)
CORS(app)

# Load the SentenceTransformer model once
model_path = "models/all-MiniLM-L6-v2"
model = SentenceTransformer(model_path)

# Initialize Pinecone index once
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("huggingface")

@app.route('/retrieve', methods=['GET'])
def retrieve():
    # Get the question from the request
    question = request.args.get('question')
    print("Got question: ", question)

    # Encode the question into a vector using SentenceTransformer
    user_vector = model.encode(question).tolist()

    response = index.query(
        namespace="ns1",
        vector=user_vector,
        top_k=4,
        include_values=True,
        include_metadata=True
    )

    # Extract and format the matched documents
    matches = response['matches']
    
    # Extract and format the score and content into a single list
    formatted_matches = [{'score': match['score'], 'content': match['metadata']['content']} for match in matches]

    print("Matches: ", formatted_matches)

    # Return the JSON response with a label
    return jsonify({'chunks': formatted_matches}), 200

if __name__ == '__main__':
    app.run(debug=True)
