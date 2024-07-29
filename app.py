import os
import gdown
import zipfile
import streamlit as st
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
import torch

# Define the URL for the zipped folder and paths
model_zip_url = 'https://drive.google.com/uc?export=download&id=1-U-oNTyHSmhnXVLk4l0seTFv0j-Ir2fT'
zip_path = 'fine-tuned-gpt2.zip'
model_dir = 'fine-tuned-gpt2'

# Download the zipped folder if it hasn't been downloaded yet
if not os.path.exists(zip_path):
    gdown.download(model_zip_url, zip_path, quiet=False)

# Unzip the folder if it hasn't been unzipped yet
if not os.path.exists(model_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(model_dir)

# Check if required files are present
expected_files = {
    'config.json',
    'model.safetensors',
    'generation_config.json',
    'vocab.json',
    'merges.txt',
    'tokenizer_config.json',
    'special_tokens_map.json'
}
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
    qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device == 'cuda' else -1)
    st.write("Model and tokenizer loaded successfully.")
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
    # Handle greetings separately
    if question.lower() in ["hi", "hello", "hey"]:
        response = "Welcome! How can I help you today?"
        st.session_state.history.append(f"Bot: {response}")
    else:
        # Add the user's question to the history
        st.session_state.history.append(f"User: {question}")

        # Create prompt with the most recent question
        prompt = f"Question: {question}\nAnswer:"
        st.write(f"Generated prompt: {prompt}")

        # Generate the bot's response
        try:
            result = qa_pipeline(prompt, max_length=100, num_return_sequences=1)
            answer = result[0]['generated_text'].strip()
            st.write(f"Generated answer: {answer}")

            # Extract the response part from the generated text
            response = answer.split('Answer:')[-1].strip() if 'Answer:' in answer else answer
            st.session_state.history.append(f"Bot: {response}")
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

# Display the conversation history with the correct alignment
for message in st.session_state.history:
    if message.startswith("User:"):
        with st.container():
            col1, col2 = st.columns([1, 6])  # Adjust the column widths as needed
            with col2:
                st.write(message)
    elif message.startswith("Bot:"):
        with st.container():
            col1, col2 = st.columns([6, 1])  # Adjust the column widths as needed
            with col1:
                st.write(message)
