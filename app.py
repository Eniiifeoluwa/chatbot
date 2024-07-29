import os
import gdown
import zipfile
import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Define the Google Drive file ID
file_id = '1-TBHbObeMZnEMu3qqL-PDUnfWLQRHQyB'
destination = 'fine-tuned-gpt2.zip'
model_dir = 'fine-tuned-gpt2'

# Download the model zip file if it doesn't exist
if not os.path.exists(model_dir):
    if not os.path.exists(destination):
        gdown.download(f'https://drive.google.com/uc?export=download&id={file_id}', destination, quiet=False)
    
    # Unzip the file
    with zipfile.ZipFile(destination, 'r') as zip_ref:
        zip_ref.extractall('.')

# Verify the model directory and files
expected_files = ['config.json', 'pytorch_model.bin', 'vocab.json', 'merges.txt', 'tokenizer_config.json', 'special_tokens_map.json']
missing_files = [ef for ef in expected_files if not os.path.exists(os.path.join(model_dir, ef))]

if missing_files:
    st.error(f"Error: The following files are missing in {model_dir}: {', '.join(missing_files)}")
    st.stop()

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the fine-tuned model and tokenizer
try:
    model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
except Exception as e:
    st.error(f"Error loading the model: {str(e)}")
    st.stop()

st.title("Restaurant Chatbot")

# Initialize session state for conversation history
if 'history' not in st.session_state:
    st.session_state.history = []

st.write("Ask me anything about our restaurant!")

question = st.text_input("Your question:")

if question:
    # Add the user's question to the history
    st.session_state.history.append(f"User: {question}")
    
    # Generate the bot's response
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs["input_ids"], max_length=100, num_return_sequences=1)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = answer.replace(prompt, "").strip()
    
    # Add the bot's answer to the history
    st.session_state.history.append(f"Bot: {answer}")

# Display the conversation history
for message in st.session_state.history:
    st.write(message)

# Greet the user if they send a greeting
if question.lower() in ["hi", "hello", "hey"]:
    st.write("Bot: Welcome! How can I help you today?")
