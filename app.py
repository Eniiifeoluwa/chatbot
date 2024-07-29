import streamlit as st
import gdown
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline

# Function to download and unzip model
def download_model(model_url, zip_filename):
    if not os.path.exists(zip_filename):
        st.write("Downloading model...")
        gdown.download(model_url, zip_filename, quiet=False)
        st.write("Download complete.")
        
        st.write("Unzipping model...")
        os.system(f"unzip {zip_filename} -d fine-tuned-gpt2")
        st.write("Unzip complete.")
    else:
        st.write("Model available to use")

# Define model URL and file names
model_url = "https://drive.google.com/uc?id=1FvoMtwUdvRIq-65sys7YuW51uo8lQE3H"  # Replace with your actual file ID
zip_filename = "fine-tuned-gpt2.zip"

# Download and unzip the model
download_model(model_url, zip_filename)

# Load the model and tokenizer
model_name = "fine-tuned-gpt2"
if not os.path.exists(model_name):
    st.write(f"Model directory {model_name} does not exist. Please check the download and unzip process.")
else:
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Create the QA pipeline
    qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

    def generate_answer(question):
        # Special case for greeting
        if question.lower() in ["hi", "hello", "hey"]:
            return "Hello! How can I assist you today? Feel free to ask about our menu or services."
        
        prompt = f"Question: {question}\nAnswer:"
        result = qa_pipeline(prompt, max_length=25, num_return_sequences=1)
        # Extract just the answer from the generated text
        answer = result[0]['generated_text'].strip()
        # Remove the prompt from the answer to prevent repeating
        return answer.replace(prompt, "").strip()

    # Streamlit app interface
    st.title("Project-3 Chat GPT")

    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Input section
    question = st.text_input("Ask a question about the restaurant:")

    if question:
        answer = generate_answer(question)
        st.session_state.history.append((question, answer))
        
    # Display history
    for q, a in st.session_state.history:
        st.write(f"**Question:** {q}")
        st.write(f"**Answer:** {a}")
        st.write("---")
