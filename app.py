import requests
from bs4 import BeautifulSoup
import nltk
import re
import numpy as np
import faiss
import os
import json
import google.generativeai as genai
from flask import Flask, request, render_template
from dotenv import load_dotenv

load_dotenv()



# Load GloVe embeddings
def load_glove_embeddings(glove_file):
    word_vectors = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            word_vectors[word] = vector
    return word_vectors

# Convert sentence to vector
def sentence_to_vector(sentence, word_vectors, dim=100):
    words = nltk.word_tokenize(sentence.lower())
    vector = np.zeros(dim)
    count = 0
    for word in words:
        if word in word_vectors:
            vector += word_vectors[word]
            count += 1
    if count != 0:
        vector /= count
    return vector

# Index text chunks with Faiss
def index_chunks(chunks, glove_file):
    word_vectors = load_glove_embeddings(glove_file)
    chunk_vectors = [sentence_to_vector(chunk, word_vectors) for chunk in chunks]
    chunk_matrix = np.array(chunk_vectors).astype('float32')
    faiss.normalize_L2(chunk_matrix)
    index = faiss.IndexFlatIP(chunk_matrix.shape[1])
    index.add(chunk_matrix)
    faiss.write_index(index, "luke_skywalker_index.faiss")
    return index, chunks

# Query Faiss index
def query_faiss(query, index, chunks, word_vectors, top_k=3):
    query_vector = sentence_to_vector(query, word_vectors).reshape(1, -1).astype('float32')
    query_vector = np.ascontiguousarray(query_vector)
    faiss.normalize_L2(query_vector)
    distances, indices = index.search(query_vector, top_k)
    return [chunks[i] for i in indices[0]]

# Make prompt for LLM
def make_prompt(query, context):
    return f"""You are a helpful and informative bot that answers questions using text from the reference passage included below.
Be sure to respond in a complete sentence, being comprehensive, including all relevant background information.
However, you are talking to a non-technical audience, so be sure to break down complicated concepts and strike a friendly and conversational tone.
If the passage is irrelevant to the answer, you may ignore it.
QUESTION: '{query}'
PASSAGE: '{context}'

ANSWER:"""

# Call Gemini API
def call_gemini_api(prompt):
    gemini_api_key = os.getenv("gemini_api_key")
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-pro')
    answer = model.generate_content(prompt)
    return answer.text

# Handle query
def handle_query(query):
    glove_file = "glove.6B.100d.txt"  # Replace with actual path
    index = faiss.read_index("luke_skywalker_index.faiss")
    chunks = json.load(open("chunks.json"))  # Load your chunks from a JSON file
    word_vectors = load_glove_embeddings(glove_file)
    context = query_faiss(query, index, chunks, word_vectors)
    prompt = make_prompt(query, " ".join(context))
    return call_gemini_api(prompt)



# Flask app setup
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    query = request.form['query']
    answer = handle_query(query)
    print(query)
    return render_template('index.html', query=query, answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
