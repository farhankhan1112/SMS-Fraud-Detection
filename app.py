import streamlit as st
import joblib
import pandas as pd

# Load model and vectorizer
model = joblib.load('sms_fraud_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Streamlit app
st.title("SMS Fraud Detection")
st.write("Enter an SMS message to classify it as **Ham** (legitimate) or **Spam** (fraudulent).")

# User input
sms_input = st.text_area("Enter SMS:", height=100)

if st.button("Predict"):
    if sms_input:
        # Transform input
        sms_tfidf = vectorizer.transform([sms_input])
        # Predict
        prediction = model.predict(sms_tfidf)[0]
        label = "Spam" if prediction == 1 else "Ham"
        # Display result
        st.write(f"**Prediction**: {label}")
        if label == "Spam":
            st.error("This message is likely fraudulent!")
        else:
            st.success("This message appears legitimate.")
    else:
        st.warning("Please enter an SMS message.")

# Example messages
st.write("### Example Messages to Try:")
st.write("- **Ham**: Hey, are you free to meet tomorrow at 5 PM?")
st.write("- **Spam**: Congratulations! You've won a $1000 gift card. Click here to claim!")