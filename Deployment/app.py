import streamlit as st
import joblib
import re
import numpy as np
from nltk.corpus import stopwords

# Load the saved components
model = joblib.load('fake_news_classifier.joblib')
port_stem = joblib.load('porter_stemmer.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [port_stem.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)


def predict(text):
    try:
        processed_text = preprocess_text(text)
        # Vectorize the text using your loaded vectorizer
        vectorized_text = vectorizer.transform([processed_text])
        
        # Get prediction probabilities
        proba = model.predict_proba(vectorized_text)[0]
        prediction = model.predict(vectorized_text)[0]
        
        # st.write(f"Prediction probabilities: {proba}")  # Debug output
        # st.write(f"Processed text: {processed_text}")
        # st.write(f"Vectorized shape: {vectorized_text.shape}")

        return prediction, proba
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

# Streamlit UI
st.title("Fake News Detector")
st.write("Enter the news content to check its authenticity")

input_text = st.text_area("News Content:", height=200)

if st.button("Analyze"):
    if input_text.strip() == "":
        st.warning("Please enter news content to analyze")
    else:
        with st.spinner("Analyzing..."):
            prediction, probabilities = predict(input_text)
            
            if probabilities is not None:
                st.subheader("Result:")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("REAL Probability", f"{probabilities[0]*100:.1f}%")
                
                with col2:
                    st.metric("FAKE Probability", f"{probabilities[1]*100:.1f}%")
                
                if prediction == 0:
                    st.success("✅ This news appears to be REAL")
                else:
                    st.error("⚠️ Warning: This news appears to be FAKE")

# Sidebar information
st.sidebar.title("About")
st.sidebar.info("""
This tool helps identify potentially fake news articles using machine learning.
It analyzes text patterns from a dataset of labeled real and fake news articles.
""")