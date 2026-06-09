import pandas as pd
import re

# Rule-based scoring
def get_risk_score(text):
    score = 0
    reasons = []

    text_lower = text.lower()

    # Urgency words
    urgency_words = ["urgent", "immediately", "action required", "act now", "limited time"]
    for word in urgency_words:
        if word in text_lower:
            score += 1
            reasons.append(f"Contains urgency word: '{word}'")

    # Account/credential bait
    account_words = ["verify your account", "confirm your password", "update your billing", "suspended", "unusual activity"]
    for word in account_words:
        if word in text_lower:
            score += 2
            reasons.append(f"Contains credential bait: '{word}'")

    # Suspicious URLs
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, text)
    for url in urls:
        if any(x in url for x in ["bit.ly", "tinyurl", "click here", ".xyz", ".tk"]):
            score += 3
            reasons.append(f"Suspicious URL found: {url}")

    # Free money bait
    money_words = ["you have won", "claim your prize", "free money", "lottery", "inheritance"]
    for word in money_words:
        if word in text_lower:
            score += 2
            reasons.append(f"Contains money bait: '{word}'")

    # Threats
    threat_words = ["your account will be closed", "legal action", "failure to respond"]
    for word in threat_words:
        if word in text_lower:
            score += 2
            reasons.append(f"Contains threat: '{word}'")

    return score, reasons


def classify(text):
    score, reasons = get_risk_score(text)
    if score >= 3:
        label = "PHISHING"
    else:
        label = "SAFE"
    return label, score, reasons