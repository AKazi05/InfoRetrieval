import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import accuracy_score

# Paths to your CSV files
test_data_path = "test_data"
files = ["resumes.csv", "academic_docs.csv", "news_articles.csv"]

# Load the CSV files into a dictionary (only first 200 rows and relevant columns)
data_frames = {}

# Relevant columns for each document type
columns = {
    'resumes.csv': 'Resume',
    'academic_docs.csv': 'ABSTRACT',
    'news_articles.csv': 'Article'
}

# Load 200 rows from each file and only the relevant columns
for file_name in files:
    file_path = os.path.join(test_data_path, file_name)
    column_name = columns[file_name]
    try:
        if file_name == 'news_articles.csv':
            data_frames[file_name] = pd.read_csv(file_path, usecols=[column_name], nrows=200, encoding='ISO-8859-1')
            print(f"Loaded {file_name} successfully with ISO-8859-1 encoding!")
        else:
            data_frames[file_name] = pd.read_csv(file_path, usecols=[column_name], nrows=200)
            print(f"Loaded {file_name} successfully!")
    except Exception as e:
        print(f"Error loading {file_name}: {e}")

# Combine all data into one DataFrame
combined_df = pd.concat(data_frames.values(), ignore_index=True)

# Create labels based on the file names
labels = []
for file_name in files:
    label = file_name.split('.')[0]
    if file_name in data_frames:
        labels.extend([label] * len(data_frames[file_name]))
    else:
        print(f"Warning: {file_name} not loaded, skipping.")
combined_df['label'] = labels

# Clean the text data
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'[^\w\s]', '', text)
        return text.lower()
    return ""

combined_df['text'] = combined_df[combined_df.columns[0]].apply(clean_text)

# Split the data into training and testing sets (80% training, 20% testing)
train_df, test_df = train_test_split(combined_df, test_size=0.2, random_state=42)

# Define valid labels
valid_labels = ['resumes', 'academic_docs', 'news_articles']

# Clean and convert labels to integers
train_labels = train_df['label'].str.strip().str.lower().apply(lambda x: valid_labels.index(x)).tolist()
test_labels = test_df['label'].str.strip().str.lower().apply(lambda x: valid_labels.index(x)).tolist()

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the text data
train_encodings = tokenizer(list(train_df['text']), truncation=True, padding=True, max_length=256)
test_encodings = tokenizer(list(test_df['text']), truncation=True, padding=True, max_length=256)

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

# Load the BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(valid_labels))

# Define a function to compute accuracy during training
def compute_metrics(p):
    preds = p.predictions.argmax(axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

# Define training arguments with memory-optimized batch size and learning rate
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,                    # increase epochs for more training time
    per_device_train_batch_size=4,         # reduced batch size for memory optimization
    per_device_eval_batch_size=8,          # reduced batch size for evaluation
    warmup_steps=500,                      # number of warmup steps
    weight_decay=0.01,                     # strength of weight decay
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,                    # slightly lower learning rate for stability with BERT
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()

# Print the evaluation results
print(eval_results)

# Save the model and tokenizer after training
model.save_pretrained("my_trained_model")
tokenizer.save_pretrained("my_trained_model")
