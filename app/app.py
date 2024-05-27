import os
from dotenv import load_dotenv, find_dotenv
from flask import Flask, request
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

load_dotenv(find_dotenv(), override=True)
pinecone_api_key = os.getenv("PINECONE_API_KEY")

app = Flask(__name__)
CORS(app)

@app.route('/retrieve', methods=['GET'])
def retrieve():
    question = request.args.get('question')
    print("Got question: ", question)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    user_vector = model.encode(question).tolist()
    print("User vector: ", user_vector)

    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index("huggingface")

    response = index.query(
        namespace="ns1",
        vector=user_vector,
        top_k=4,
        include_values=True,
        include_metadata=True
    )

    print("Response: ", response)

    matches = response['matches']
    documents = [match['metadata']['content'] for match in matches] 

    formatted_documents = ""

    for i, doc in enumerate(documents, 1):
        formatted_documents += f"Chunk Reference {i}: {doc}\n"

    return formatted_documents, 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
