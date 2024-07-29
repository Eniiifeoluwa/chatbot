import os
import gdown
import streamlit as st
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
import torch

# Define the directory and file IDs
model_dir = 'fine-tuned-gpt2'
file_ids = {
    'config.json': '1-LpqdLnsaw3fp_KQC4-TiH2Ir4Hql-eE',
    'model.safetensors': '1-S2AIPpo7L4k2gZKI-FMDCawKDTmj9b-',
    'generation_config.json': '1-3zQfTXbmuxyxpenqVR80GIDTGvSY62o',
    'vocab.json': '1-NqPU2oIFxORmTPDiwoiamlCu6Mb59vj',
    'merges.txt': '1-45d87LI3k03DN0Umsc7tePl33Y0b5Wb',
    'tokenizer_config.json': '1-8fcX7ePKW7Hc-2r9ZKZmN6deiq4S03a',
    'special_tokens_map.json': '1-Sq8XlcVxfutKTbal0idqKH5qBHx1Z9a',
}

# Create the model directory if it doesn't exist
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Download each file
for filename, file_id in file_ids.items():
    file_path = os.path.join(model_dir, filename)
    if not os.path.exists(file_path):
        gdown.download(f'https://drive.google.com/uc?export=download&id={file_id}', file_path, quiet=False)

# Check if required files are present
expected_files = file_ids.keys()
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
        prompt = f"User: {question}\nBot:"

        # Generate the bot's response
        result = qa_pipeline(prompt, max_length=100, num_return_sequences=1)
        answer = result[0]['generated_text'].strip()

        # Extract the response after 'Bot:' in the answer
        response = answer.split("Bot:")[-1].strip()
        
        # Add the bot's answer to the history
        st.session_state.history.append(f"Bot: {response}")

# Display the conversation history
for message in st.session_state.history:
    st.write(message)
