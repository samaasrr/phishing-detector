import streamlit as st
from detector import classify
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

st.set_page_config(page_title="Phishing Detector", layout="centered")

st.markdown("""
    <style>
        .stTextArea textarea {
            background-color: #1a1a1a;
            color: #ffffff;
            border: 1px solid #333333;
            border-radius: 6px;
            font-size: 14px;
        }
        .stButton button {
            background-color: #ffffff;
            color: #000000;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            padding: 0.5rem 2rem;
            width: 100%;
        }
        .stButton button:hover {
            background-color: #e0e0e0;
        }
        .result-card {
            background-color: #1a1a1a;
            border-radius: 8px;
            padding: 20px 24px;
            margin-top: 8px;
            border: 1px solid #2a2a2a;
        }
        .result-label {
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 6px;
        }
        .result-sub {
            font-size: 13px;
            color: #888888;
            margin-bottom: 12px;
        }
        .reason {
            font-size: 13px;
            color: #aaaaaa;
            margin: 3px 0;
        }
        .col-header {
            font-size: 12px;
            font-weight: 600;
            color: #666666;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 8px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("## Phishing Email Detector")
st.markdown("<p style='color:#888888; margin-top:-8px; margin-bottom:24px;'>Paste an email to analyze it against a rule-based scorer and a fine-tuned DistilBERT model.</p>", unsafe_allow_html=True)

email_input = st.text_area("", height=180, placeholder="Paste email content here...")

@st.cache_resource
def load_ml_model():
    tokenizer = DistilBertTokenizer.from_pretrained('./model')
    model = DistilBertForSequenceClassification.from_pretrained('./model')
    model.eval()
    return tokenizer, model

def ml_predict(text):
    tokenizer, model = load_ml_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()
    return "PHISHING" if pred == 1 else "SAFE", round(confidence * 100, 2)

if st.button("Analyze"):
    if email_input.strip() == "":
        st.warning("Paste some email content first.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='col-header'>Rule-based</div>", unsafe_allow_html=True)
            label, score, reasons = classify(email_input)
            color = "#ff5555" if label == "PHISHING" else "#50fa7b"
            reasons_html = "".join([f"<div class='reason'>- {r}</div>" for r in reasons]) if reasons else "<div class='reason'>No flags found.</div>"
            st.markdown(f"""
                <div class='result-card'>
                    <div class='result-label' style='color:{color};'>{label}</div>
                    <div class='result-sub'>Risk score: {score}</div>
                    {reasons_html}
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='col-header'>DistilBERT</div>", unsafe_allow_html=True)
            with st.spinner("Running model..."):
                ml_label, confidence = ml_predict(email_input)
            color = "#ff5555" if ml_label == "PHISHING" else "#50fa7b"
            st.markdown(f"""
                <div class='result-card'>
                    <div class='result-label' style='color:{color};'>{ml_label}</div>
                    <div class='result-sub'>Confidence: {confidence}%</div>
                    <div class='reason' style='margin-top:12px;'>82k+ emails &nbsp;·&nbsp; 95% accuracy &nbsp;·&nbsp; 0.99 precision</div>
                </div>
            """, unsafe_allow_html=True)