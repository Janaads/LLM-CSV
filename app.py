import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables from .env file
load_dotenv()

# Path to your CSV file
csv_file_path = 'csv\sample 1.csv'

# Load CSV data and precompute embeddings
def load_data_and_embeddings(file_path):
    data = pd.read_csv(file_path)
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    embeddings = model.encode(data.astype(str).values.tolist(), convert_to_tensor=True, show_progress_bar=True)
    return data, embeddings, model

# Function to find the most relevant rows based on the query
def find_relevant_rows(query, data, embeddings, model, top_k=5):
    query_embedding = model.encode([query], convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k].cpu().numpy()
    return data.iloc[top_results]

# Function to chat with CSV data
def chat_with_csv(df, query, embeddings, model):
    relevant_rows = find_relevant_rows(query, df, embeddings, model)
    
    # Prepare the context for the LLM based on the relevant rows
    context = relevant_rows.to_string(index=False)
    
    # Load the GROQ API key from environment variables
    groq_api_key = os.environ['GROQ_API_KEY']
    
    # Initialize GROQ chat with provided API key, model name, and settings
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192", temperature=0.2)
    
    # Prepare messages for the chat model
    messages = [
        SystemMessage(content="You are a helpful assistant. Use the provided context to answer the user's question in a concise way, aiming for 2 to 3 sentences."),
        HumanMessage(content=f"Context: {context}\n\nQuestion: {query}")
    ]
    
    # Chat with the context
    result = llm(messages)
    return result.content

# Set layout configuration for the Streamlit page
st.set_page_config(layout='wide')
st.title("LLM CSV Chat")

# Load data and embeddings from the specified CSV file
try:
    data, embeddings, model = load_data_and_embeddings(csv_file_path)
    st.info("CSV loaded successfully")
    st.dataframe(data.head(3), use_container_width=True)

    # Enter the query for analysis
    input_text = st.text_area("Enter the query")

    # "Chat with CSV" button is always available
    if st.button("Chat with CSV"):
        if input_text:
            st.info("Your Query: " + input_text)
            result = chat_with_csv(data, input_text, embeddings, model)
            st.success(result)
        else:
            st.warning("Please enter a query before chatting.")
except Exception as e:
    st.error(f"Error loading data or embeddings: {e}")
