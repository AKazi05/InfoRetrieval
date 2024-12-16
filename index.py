import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

test_data_path = "test_data"
files = ["resumes.csv", "academic_docs.csv", "news_articles.csv"]

data_frames = {}

columns = {
    'resumes.csv': 'Resume',
    'academic_docs.csv': 'ABSTRACT',
    'news_articles.csv': 'Article'
}

for file_name in files:
    file_path = os.path.join(test_data_path, file_name)
    column_name = columns[file_name]
    try:
        if file_name == 'news_articles.csv':
            data_frames[file_name] = pd.read_csv(file_path, usecols=[column_name], nrows=2500, encoding='ISO-8859-1')
            print(f"Loaded {file_name} successfully with ISO-8859-1 encoding!")
        else:
            data_frames[file_name] = pd.read_csv(file_path, usecols=[column_name], nrows=2500)
            print(f"Loaded {file_name} successfully!")
    except Exception as e:
        print(f"Error loading {file_name}: {e}")

combined_df = pd.concat(data_frames.values(), ignore_index=True)

labels = []
for file_name in files:
    label = file_name.split('.')[0]
    if file_name in data_frames:
        labels.extend([label] * len(data_frames[file_name]))
    else:
        print(f"Warning: {file_name} not loaded, skipping.")
combined_df['label'] = labels

def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'[^\w\s]', '', text)
        return text.lower()
    return ""

combined_df['text'] = combined_df[combined_df.columns[0]].apply(clean_text)

train_df, test_df = train_test_split(combined_df, test_size=0.2, random_state=42)

valid_labels = ['resumes', 'academic_docs', 'news_articles']

train_labels = train_df['label'].str.strip().str.lower().apply(lambda x: valid_labels.index(x)).tolist()
test_labels = test_df['label'].str.strip().str.lower().apply(lambda x: valid_labels.index(x)).tolist()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(list(train_df['text']), truncation=True, padding=True, max_length=256)
test_encodings = tokenizer(list(test_df['text']), truncation=True, padding=True, max_length=256)

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

train_dataset = TextDataset(train_encodings, train_labels)
test_dataset = TextDataset(test_encodings, test_labels)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(valid_labels))

def compute_metrics(p):
    preds = p.predictions.argmax(axis=1)
    accuracy = accuracy_score(p.label_ids, preds)
    precision = precision_score(p.label_ids, preds, average='weighted')
    recall = recall_score(p.label_ids, preds, average='weighted')
    f1 = f1_score(p.label_ids, preds, average='weighted')
    class_report = classification_report(p.label_ids, preds, output_dict=True)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": class_report
    }

training_args = TrainingArguments(
    output_dir='./results',                    
    num_train_epochs=3,                        
    per_device_train_batch_size=16,            
    per_device_eval_batch_size=32,             
    warmup_steps=1000,                         
    weight_decay=0.01,                         
    logging_dir='./logs',                      
    logging_steps=50,                          
    eval_strategy="epoch",                     
    save_strategy="epoch",                     
    learning_rate=2e-5,                        
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

eval_results = trainer.evaluate()

print(eval_results)

model.save_pretrained("my_trained_model")
tokenizer.save_pretrained("my_trained_model")