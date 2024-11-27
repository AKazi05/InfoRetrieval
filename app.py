from flask import Flask, render_template, request, redirect, url_for
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from werkzeug.utils import secure_filename
import os
from PyPDF2 import PdfReader
from docx import Document
import re
import spacy

app = Flask(__name__)

# Ensure the uploads directory exists
os.makedirs("./uploads", exist_ok=True)

# Load the model and tokenizer
model_path = "InfoRetrieval-2\my_trained_model" #change this to relative path of the model
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Allowed file types
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def file_to_text(file_path):
    """Converts file content into text and extracts the abstract for academic documents."""
    ext = file_path.rsplit('.', 1)[-1].lower()
    
    # Process PDF files
    if ext == 'pdf':
        reader = PdfReader(file_path)
        full_text = " ".join(page.extract_text() for page in reader.pages)
    
    # Process DOCX files
    elif ext == 'docx':
        doc = Document(file_path)
        full_text = " ".join(paragraph.text for paragraph in doc.paragraphs)
    
    # Process plain text files
    elif ext == 'txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
    
    # For DOC files (requires LibreOffice or similar)
    elif ext == 'doc':
        try:
            # Use LibreOffice/command-line conversion to docx
            converted_path = file_path + ".docx"
            os.system(f"soffice --headless --convert-to docx \"{file_path}\" --outdir ./uploads")
            doc = Document(converted_path)
            os.remove(converted_path)  # Clean up the converted file
            full_text = " ".join(paragraph.text for paragraph in doc.paragraphs)
        except Exception as e:
            raise ValueError(f"Error processing DOC file: {e}")
    
    else:
        raise ValueError("Unsupported file type")

    # Extract the abstract (case insensitive)
    abstract = extract_abstract(full_text)
    return abstract or full_text  # Fallback to full text if abstract is not found

def extract_abstract(text):
    """Extracts the abstract section from the text."""
    import re

    # Common abstract header patterns
    abstract_patterns = [
        r"(?<=\n)abstract[:\s]*(.*?)(?=\n\w|$)",  # Match "Abstract" section
        r"(?<=\n)summary[:\s]*(.*?)(?=\n\w|$)",   # Match "Summary" (alternate term)
    ]

    for pattern in abstract_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()

    return None  

def predict_category(text):
    """Predicts the category of the document."""

    if "news" in text.lower():
        return "News Article"

    # Check explicitly for keywords indicating Academic Documents
    if any(keyword in text.lower() for keyword in ["abstract", "a b s t r a c t"]):
        return "Academic Document"

    # Tokenize and format the text for BERT
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()

    # Map class IDs to labels
    labels = {0: "Resume", 2: "Academic Document", 1: "News Article"}
    return labels.get(predicted_class_id, "News Article")  # Default to News Article if no valid prediction

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # Check if the post request has the file part
        if "file" not in request.files:
            return render_template("upload.html", error="No file uploaded!")

        file = request.files["file"]

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = f"./uploads/{filename}"
            file.save(file_path)

            try:
                # Convert file to text
                text = file_to_text(file_path)

                # Predict category
                prediction = predict_category(text)

                return render_template("result.html", prediction=prediction)
            except Exception as e:
                return render_template("upload.html", error=f"Error processing file: {e}")
        else:
            return render_template("upload.html", error="Invalid file type!")

    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)