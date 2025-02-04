# app.py
import os
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import subprocess
from pyngrok import ngrok

# Initialize the models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# File paths for FAISS index and chunks
faiss_file = "legal_index.faiss"
chunks_file = "chunks.txt"

# Check if required files exist
def check_files():
    if not os.path.exists(faiss_file) or not os.path.exists(chunks_file):
        return False
    return True

# Load FAISS index and chunks (ensure these are generated beforehand)
def load_files():
    if not check_files():
        return False, None, None
    index = faiss.read_index(faiss_file)
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = f.read().splitlines()
    return True, index, chunks

def retrieve_top_chunks(query, index, chunks, k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    results = [chunks[i] for i in indices[0]]
    return results

def generate_answer(query, context):
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = generator(prompt, max_length=200, num_return_sequences=1)
    return response[0]["generated_text"]

def process_query(query, index, chunks):
    try:
        # Retrieve top-k chunks
        top_chunks = retrieve_top_chunks(query, index, chunks, k=5)
        context = "\n\n".join(top_chunks)
        # Generate answer
        answer = generate_answer(query, context)
        return answer,top_chunks
    except Exception as e:
        return "Error processing query.", str(e)

def main():
    # Check if the necessary files exist
    files_exist, index, chunks = load_files()

    if not files_exist:
        st.error("Required files (legal_index.faiss and chunks.txt) are missing. Please upload or generate them first.")
        return

    # Initialize session state for navigation
    if "page" not in st.session_state:
        st.session_state.page = "welcome"
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Custom CSS for styling
    st.markdown("""
        <style>
            .user-msg {
                background-color: #d1c4e9;
                color: #006064;
                padding: 10px;
                border-radius: 10px;
                margin: 10px 0;
                max-width: 80%;
                margin-left: 20%;
            }
            .bot-msg {
                background-color: #f3e5f5;
                color: #bf360c;
                padding: 10px;
                border-radius: 10px;
                margin: 10px 0;
                max-width: 80%;
            }
            .welcome-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 80vh;
                text-align: center;
                background-color: #6a4c93;
                color: white;
                padding: 2rem;
                border-radius: 10px;
            }
            .chat-button {
                background-color: #8e4e99;
                color: white;
                padding: 0.8rem 2rem;
                border-radius: 5px;
                border: none;
                font-size: 1.2rem;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            .chat-button:hover {
                background-color: #7a3b84;
            }
        </style>
    """, unsafe_allow_html=True)

    # Welcome Page
    if st.session_state.page == "welcome":
        with st.container():
            st.markdown("""
                <div class="welcome-container">
                    <h1 style="font-size: 4rem; margin-bottom: 1rem;">JURIS.ai</h1>
                    <p style="font-size: 1.5rem; margin-bottom: 2rem;">A legal Q&A system powered by AI.</p>
                </div>
            """, unsafe_allow_html=True)
            
            if st.button("Start Chat", key="start_chat", use_container_width=True):
                st.session_state.page = "chat"
                st.rerun()

    # Chat Page
    elif st.session_state.page == "chat":
        st.title("Legal Q&A Chatbot")
        
        # Chat container
        chat_container = st.container()
        
        # Input container at the bottom
        input_container = st.container()
        
        with chat_container:
            for msg in st.session_state.messages:
                if msg["sender"] == "user":
                    st.markdown(f'<div class="user-msg">{msg["message"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="bot-msg">{msg["message"]}</div>', unsafe_allow_html=True)

        with input_container:
            # Use a unique key for each input based on session state length
            user_input = st.text_input("Enter your legal question:", key=f"user_input_{len(st.session_state.messages)}")
            
            if user_input:
                # Add user input to the message list
                st.session_state.messages.append({"sender": "user", "message": user_input})
                
                with st.spinner("Generating answer..."):
                    # Process query
                    answer, context = process_query(user_input, index, chunks)
                
                # Add bot's response
                st.session_state.messages.append({
                    "sender": "bot", 
                    "message": f"Answer: {answer}\n\n*Context:* {context}"
                })

                # Clear input (without rerun)
                st.text_input("Enter your legal question:", key=f"user_input_{len(st.session_state.messages) + 1}", value="")
                st.rerun()

if __name__ == "__main__":
    main()
