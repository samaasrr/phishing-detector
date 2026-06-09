import streamlit as st
from detector import classify

st.set_page_config(page_title="Phishing Detector")

st.title("Phishing Email Detector")
st.write("Paste an email below to check if it's phishing or safe.")

email_input = st.text_area("Email Content", height=200, placeholder="Paste email text here...")

if st.button("Analyze"):
    if email_input.strip() == "":
        st.warning("Please paste some email content first!")
    else:
        label, score, reasons = classify(email_input)

        if label == "PHISHING":
            st.error(f"PHISHING DETECTED - Risk Score: {score}")
        else:
            st.success(f"SAFE - Risk Score: {score}")

        if reasons:
            st.subheader("Why it was flagged:")
            for reason in reasons:
                st.write(f"- {reason}")
        else:
            st.write("No suspicious patterns found.")