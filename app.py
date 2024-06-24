import os
import json
import gc
import logging
from dotenv import load_dotenv, find_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from pinecone.grpc import PineconeGRPC as test

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(find_dotenv(), override=True)
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Initialize Flask app and CORS
app = Flask(__name__)
CORS(app)

# Lazy loading the SentenceTransformer model
model = None

def get_model():
    global model
    if model is None:
        logger.info("Loading the SentenceTransformer model...")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model

# Initialize Pinecone and index once at startup
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("huggingface")

@app.route('/namespace', methods=['GET'])
def list_namespaces():
    try:
        logger.info("Listing all namespaces")
        index = pc.Index("huggingface")
        namespaces = index.describe_index_stats()
        namespace_names = list(namespaces['namespaces'].keys())
        print(namespace_names)
        return jsonify({'namespaces': namespace_names}), 200
    except Exception as e:
        logger.error(f"An error occurred while listing namespaces: {e}")
        return jsonify({'error': 'An error occurred while listing namespaces, please try again later'}), 500

@app.route('/retrieve', methods=['GET'])
def retrieve():
    try:
        # Get the question and namespace from the request
        question = request.args.get('question')
        namespace = request.args.get('namespace')
        k = request.args.get('k', 4)
        logger.info(f"Got question: {question}")
        logger.info(f"Got namespace: {namespace}")
        logger.info(f"Got k: {k}")

        # Validate namespace
        if not namespace:
            logger.error("Namespace is required.")
            return jsonify({'error': 'Namespace is required'}), 400

        # Strip unwanted characters and convert to int
        try:
            k = int(k.strip())
        except ValueError:
            k = 4  # Default value if conversion fails
            logger.warning(f"Invalid value for k, defaulting to {k}")

        # Encode the question into a vector using the preloaded SentenceTransformer model
        model = get_model()
        user_vector = model.encode(question).tolist()

        # Query the Pinecone index with the dynamic namespace
        response = index.query(
            namespace=namespace,
            vector=user_vector,
            top_k=k,
            include_values=True,
            include_metadata=True
        )

        # Extract and format the matched documents
        matches = response['matches']

        # Extract and format the score and content into a single list
        formatted_matches = [{'score': match['score'], 'content': match['metadata']['content']} for match in matches]

        # Manual garbage collection
        gc.collect()

        # Return the JSON response with a label
        return jsonify({'chunks': formatted_matches}), 200
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return jsonify({'error': 'An error occurred, please try again later'}), 500

if __name__ == '__main__':
    app.run(debug=True)
