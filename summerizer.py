import os
import docx
from PyPDF2 import PdfReader
from transformers import pipeline
import tensorflow as tf
from tqdm import tqdm  # To show progress during chunk processing

# Check if GPU is available for TensorFlow
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Load summarizer model in the main process
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if tf.config.experimental.list_physical_devices('GPU') else -1)  # Use GPU if available

# Summarization function using Hugging Face Transformers
def summarize(text):
    try:
        input_length = len(text.split())  # Calculate the length of the input text in words
        max_length = min(150, input_length)  # Set max_length based on the input length

        # Perform summarization with proper max/min length constraints
        summary = summarizer(text, max_length=max_length, min_length=50, do_sample=False)
        output_string = summary[0]['summary_text']
    except Exception as e:
        # Fallback if summarization fails
        output_string = text[:150] + "..."  # Truncate as a backup
    return output_string

# Summarize chunks of text in parallel
def summarize_chunks(text, chunk_size):
    words_in_text = text.split()
    count = 0
    chunk = ""
    chunks = []

    for word in words_in_text:
        if count >= chunk_size:
            count = 0
            chunks.append(chunk.strip())
            chunk = ""
        chunk += word + " "
        count += 1

    if chunk:  # Append any remaining text
        chunks.append(chunk.strip())

    # Summarize each chunk (use tqdm for progress tracking)
    summarized_string = ""
    for chunk in tqdm(chunks, desc="Summarizing Chunks", unit="chunk"):
        summarized_string += summarize(chunk) + "\n"

    return summarized_string

# Extract text from supported file types
def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return text

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Process files based on their extensions
def process_file(file_path, chunk_size):
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    if file_extension == '.docx':
        text = extract_text_from_docx(file_path)
    elif file_extension == '.pdf':
        text = extract_text_from_pdf(file_path)
    elif file_extension == '.txt':
        text = extract_text_from_txt(file_path)
    else:
        raise ValueError("Unsupported file type. Only .docx, .pdf, and .txt are allowed.")

    return summarize_chunks(text, chunk_size)

# Main entry point for testing
if __name__ == "__main__":
    try:
        # Prompt the user for a file path
        file_path = input("Enter the path to your text file (.txt, .docx, .pdf): ").strip()
        if not os.path.exists(file_path):
            print("Error: The file does not exist. Please check the path and try again.")
        else:
            chunk_size = int(input("Enter the chunk size (number of words per chunk): "))
            summarized_output = process_file(file_path, chunk_size)
            print("\nSummarized Output:\n")
            print(summarized_output)
    except Exception as e:
        print(f"An error occurred: {e}")
