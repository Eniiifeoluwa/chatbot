import os
import gdown
import zipfile
import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch


file_id = '1-TBHbObeMZnEMu3qqL-PDUnfWLQRHQyB'
destination = 'fine-tuned-gpt2.zip'


if not os.path.exists('fine-tuned-gpt2'):
    if not os.path.exists(destination):
        gdown.download(f'https://drive.google.com/uc?export=download&id={file_id}', destination, quiet=False)
    
    # Unzip the file
    with zipfile.ZipFile(destination, 'r') as zip_ref:
        zip_ref.extractall('.')


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained("./fine-tuned-gpt2").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("./fine-tuned-gpt2")

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
