import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch

# Load the data
print("Loading dataset...")
df = pd.read_csv('phishing_email.csv')
df = df.dropna()
df = df.sample(5000, random_state=42)

# Same split as training
_, X_test, _, y_test = train_test_split(
    df['text_combined'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42
)

# Load the saved model
print("Loading model...")
tokenizer = DistilBertTokenizer.from_pretrained('./model')
model = DistilBertForSequenceClassification.from_pretrained('./model')
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# the dataset
class EmailDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=256)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

print("Tokenizing...")
test_dataset = EmailDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=16)

# eveluating 
print("Evaluating...")
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\n--- Evaluation Results ---")
print(classification_report(all_labels, all_preds, target_names=["Safe", "Phishing"]))