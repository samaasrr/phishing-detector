# Phishing Email Detector

A simple yet very effective phishing email classifier built in two stages — a rule-based scoring system and a fine-tuned DistilBERT model trained on 82,000+ labeled emails.

## How it works

The app runs two detectors side by side:

- **Rule-based**: scores emails based on patterns like urgency keywords, credential bait, and suspicious URLs
- **DistilBERT**: a transformer model fine-tuned on the dataset, achieving 95% accuracy and 0.99 phishing precision on the test set

## Results

| Metric | Safe | Phishing |
|--------|------|----------|
| Precision | 0.91 | 0.99 |
| Recall | 0.99 | 0.92 |
| F1 | 0.95 | 0.95 |

## Stack

- Python, PyTorch, Hugging Face Transformers
- Streamlit for the UI

## If you want to Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
phishing-detector/
├── app.py          # Streamlit UI (Still new to this)
├── detector.py     # Rule-based scorer
├── train.py        # DistilBERT fine-tuning
├── evaluate.py     # Precision, recall, F1 evaluation
```
