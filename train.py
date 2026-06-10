import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch
from torch.optim import AdamW

# Load data
print("Loading dataset...")
df = pd.read_csv('phishing_email.csv')
df = df.dropna()
df = df.sample(5000, random_state=42)  # use 5000 rows to keep it fast

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df['text_combined'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


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
train_dataset = EmailDataset(X_train, y_train)
test_dataset = EmailDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

print("Loading model...")
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
optimizer = AdamW(model.parameters(), lr=2e-5)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)

print("Training...")
model.train()
for epoch in range(2):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

# Save model and print the
print("Saving model...")
model.save_pretrained('./model')
tokenizer.save_pretrained('./model')
print("Done! Model saved to ./model")