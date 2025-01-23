import streamlit as st
from transformers import pipeline

# Title for the app
st.title("Sentiment Classification with BERT")

# Load the sentiment analysis model using PyTorch (framework="pt")
classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english', framework="pt")

# Input text area
text = st.text_area("Enter Your Text Here")

# Predict button
if st.button("Predict"):
    if text.strip():  # Check if text is not empty
        result = classifier(text)  # Perform prediction
        st.write("Prediction Result:", result)  # Display result
    else:
        st.write("Please enter some text for analysis.")

      