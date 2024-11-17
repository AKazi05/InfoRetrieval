from flask import Flask, render_template, request, redirect, url_for
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from werkzeug.utils import secure_filename
import os
from PyPDF2 import PdfReader
from docx import Document

app = Flask(__name__)

# Ensure the uploads directory exists
os.makedirs("./uploads", exist_ok=True)

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained("my_trained_model")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Allowed file types
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def file_to_text(file_path):
    """Converts file content into text based on its type."""
    ext = file_path.rsplit('.', 1)[-1].lower()
    
    # Process PDF files
    if ext == 'pdf':
        reader = PdfReader(file_path)
        return " ".join(page.extract_text() for page in reader.pages)
    
    # Process DOCX files
    elif ext == 'docx':
        doc = Document(file_path)
        return " ".join(paragraph.text for paragraph in doc.paragraphs)
    
    # Process plain text files
    elif ext == 'txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    # For DOC files (requires LibreOffice or similar)
    elif ext == 'doc':
        try:
            # Use LibreOffice/command-line conversion to docx
            converted_path = file_path + ".docx"
            os.system(f"soffice --headless --convert-to docx \"{file_path}\" --outdir ./uploads")
            doc = Document(converted_path)
            os.remove(converted_path)  # Clean up the converted file
            return " ".join(paragraph.text for paragraph in doc.paragraphs)
        except Exception as e:
            raise ValueError(f"Error processing DOC file: {e}")
    
    else:
        raise ValueError("Unsupported file type")

def predict_category(text):
    """Predicts the category of the document."""
    # Tokenize and format the text for BERT
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()

    # Map class IDs to labels
    labels = {0: "Resume", 1: "Academic Document", 2: "News Article"}
    return labels[predicted_class_id]

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

                # Clean up the uploaded file if needed
                # os.remove(file_path)  # Uncomment if you want to delete after prediction

                return render_template("result.html", prediction=prediction)
            except Exception as e:
                return render_template("upload.html", error=f"Error processing file: {e}")
        else:
            return render_template("upload.html", error="Invalid file type!")

    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)