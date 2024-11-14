import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import accuracy_score

# Paths to your CSV files
test_data_path = "test_data"
files = ["resumes.csv", "academic_docs.csv", "emails.csv", "news_articles.csv"]

# Load the CSV files into a dictionary (only first 100 rows and relevant columns)
data_frames = {}

# Relevant columns for each document type
columns = {
    'resumes.csv': 'Resume',
    'academic_docs.csv': 'abstract',
    'emails.csv': 'message',
    'news_articles.csv': 'Article'
}

# Load only the relevant columns and first 100 rows from each file
for file_name in files:
    file_path = os.path.join(test_data_path, file_name)
    column_name = columns[file_name]
    try:
        data_frames[file_name] = pd.read_csv(file_path, usecols=[column_name], nrows=100)
        print(f"Loaded {file_name} successfully!")
    except Exception as e:
        print(f"Error loading {file_name}: {e}")

# Combine all data into one DataFrame
combined_df = pd.concat(data_frames.values(), ignore_index=True)

# Create labels based on the file names (this will map each document to its corresponding category)
labels = []
for file_name in files:
    label = file_name.split('.')[0]  # Use the file name without extension as the label
    if file_name in data_frames:  # Only add labels for files that were successfully loaded
        labels.extend([label] * len(data_frames[file_name]))
    else:
        print(f"Warning: {file_name} not loaded, skipping.")
combined_df['label'] = labels

# Clean the text data by removing non-alphanumeric characters
def clean_text(text):
    if isinstance(text, str):  # Only process text if it's a string
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and special characters
        return text.lower()  # Convert to lowercase
    return ""

combined_df['text'] = combined_df[combined_df.columns[0]].apply(clean_text)  # Apply cleaning function to the text column

# Split the data into training and testing sets (80% training, 20% testing)
train_df, test_df = train_test_split(combined_df, test_size=0.2, random_state=42)

# Define valid labels (this should match the keys of your `file_details`)
valid_labels = ['resumes', 'academic_docs', 'emails', 'news_articles']  # Labels for your categories

# Clean the labels by stripping any extra spaces and converting to lowercase
train_df['label'] = train_df['label'].str.strip()  # Remove leading/trailing spaces
train_df['label'] = train_df['label'].str.lower()  # Convert all labels to lowercase for consistency

# Convert labels to integers using the valid labels
def get_label_index(label):
    if label in valid_labels:
        return valid_labels.index(label)
    else:
        return -1  # Return -1 for labels that are not found (optional: handle these later)

train_labels = train_df['label'].apply(get_label_index).tolist()
test_labels = test_df['label'].apply(get_label_index).tolist()

# Check if any labels were not found
if -1 in train_labels or -1 in test_labels:
    print("Warning: Some labels were not found in valid labels.")
    train_labels = [label if label != -1 else 0 for label in train_labels]  # Assigning default value of 0 for invalid labels
    test_labels = [label if label != -1 else 0 for label in test_labels]

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the text data
train_encodings = tokenizer(list(train_df['text']), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(list(test_df['text']), truncation=True, padding=True, max_length=512)

# Prepare the datasets for training (convert to torch tensors)
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create train and test datasets
train_dataset = TextDataset(train_encodings, train_labels)
test_dataset = TextDataset(test_encodings, test_labels)

# Print the length of the datasets
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")

# Load pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(valid_labels))

# Define a function to compute accuracy during training
def compute_metrics(p):
    preds = p.predictions.argmax(axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

# Define training arguments with updated argument name
training_args = TrainingArguments(
    output_dir='./results',          # output directory for results
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=16,  # batch size for training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    eval_strategy="epoch",           # updated evaluation strategy
    save_strategy="epoch",           # save strategy
)

# Initialize the Trainer with the model, training arguments, datasets, and metrics function
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,           # evaluation dataset
    compute_metrics=compute_metrics,     # metrics function
)

# Start training
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()

# Print the evaluation results (accuracy, etc.)
print(eval_results)
