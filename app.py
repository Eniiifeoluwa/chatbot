import os
import torch
import gdown
import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline

# Define paths
model_path = 'fine-tuned-gpt2.pth'
tokenizer_dir = './fine-tuned-gpt2'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Download the model if not present
if not os.path.exists(model_path):
    model_pth_url = 'https://drive.google.com/uc?export=download&id=10J4BzyV6uxOFkXbsoHqoB3qapS63DAtn'
    gdown.download(model_pth_url, model_path, quiet=False)

# Check if the tokenizer directory exists
if not os.path.exists(tokenizer_dir):
    st.error("Tokenizer directory is missing. Please check the path.")
    st.stop()

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_dir)

# Load the model and its state dictionary
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.load_state_dict(torch.load(model_path))
model.to(device)

# Initialize the text generation pipeline
qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device == 'cuda' else -1)

st.title("Restaurant Chatbot")

# Initialize session state for conversation history
if 'history' not in st.session_state:
    st.session_state.history = []

st.write("Ask me anything about our restaurant!")

question = st.text_input("Your question:")

if question:
    # Handle greetings separately
    if question.lower() in ["hi", "hello", "hey"]:
        response = "Welcome! How can I help you today?"
        st.session_state.history.append(f"Bot: {response}")
    else:
        # Add the user's question to the history
        st.session_state.history.append(f"User: {question}")

        # Create prompt with the most recent question
        prompt = f"Question: {question}\nAnswer:"

        # Generate the bot's response
        result = qa_pipeline(prompt, max_length=100, num_return_sequences=1)
        answer = result[0]['generated_text'].strip()

        # Extract the response part from the generated text
        if 'Answer:' in answer:
            response = answer.split('Answer:')[-1].strip()
        else:
            response = answer

        # Add the bot's answer to the history
        st.session_state.history.append(f"Bot: {response}")

# Display the conversation history with messages aligned
for message in st.session_state.history:
    if message.startswith("User:"):
        st.write(f"<div style='text-align: right;'>{message}</div>", unsafe_allow_html=True)
    else:
        st.write(f"<div style='text-align: left;'>{message}</div>", unsafe_allow_html=True)
