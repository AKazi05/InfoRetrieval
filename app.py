from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from werkzeug.utils import secure_filename
import os
from PyPDF2 import PdfReader
from docx import Document
import nltk
from tqdm import tqdm

app = Flask(__name__)

os.makedirs("./uploads", exist_ok=True)

classifier_model = BertForSequenceClassification.from_pretrained("my_trained_model")
# Load the model and tokenizer
model_path = "InfoRetrieval-2\my_trained_model" #change this to relative path of the model
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_category(text):

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    outputs = classifier_model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()

    labels = {0: "Resume", 1: "News Article", 2: "Academic Document"}

def file_to_text(file_path):
    """Converts file content into text."""
    ext = file_path.rsplit('.', 1)[-1].lower()
    if ext == 'pdf':
        reader = PdfReader(file_path)
        return " ".join(page.extract_text() for page in reader.pages)
    elif ext == 'docx':
        doc = Document(file_path)
        return " ".join(paragraph.text for paragraph in doc.paragraphs)
    elif ext == 'txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError("Unsupported file type")

def summarize(text):
    try:
        input_length = len(text.split())
        max_length = min(150, input_length)
        summary = summarizer(text, max_length=max_length, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    except Exception:
        return text[:150] + "..."  
    
def summarize_chunks(text, chunk_size):
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return "\n".join([summarize(chunk) for chunk in tqdm(chunks, desc="Summarizing chunks")])

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("upload.html", error="No file uploaded!")
        file = request.files["file"]
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = f"./uploads/{filename}"
            file.save(file_path)
            try:
                text = file_to_text(file_path)
                prediction = predict_category(text)
                return render_template("result.html", prediction=prediction)
            except Exception as e:
                return render_template("upload.html", error=f"Error processing file: {e}")
        else:
            return render_template("upload.html", error="Invalid file type!")
    return render_template("upload.html")

@app.route("/summarize", methods=["GET", "POST"])
def upload_summarizer_file():
    if request.method == "POST":
        if "summary-file" not in request.files:
            return render_template("upload.html", error="No file uploaded for summarization!")
        file = request.files["summary-file"]
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = f"./uploads/{filename}"
            file.save(file_path)
            try:
                text = file_to_text(file_path)
                summarized_text = summarize_chunks(text, chunk_size=350)  # Adjust chunk size as needed
                return render_template("result2.html", summary=summarized_text)
            except Exception as e:
                return render_template("upload.html", error=f"Error processing file: {e}")
        else:
            return render_template("upload.html", error="Invalid file type!")
    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)