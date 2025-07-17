import streamlit as st
import nltk
from nltk import NaiveBayesClassifier
from joblib import load

# Download NLTK resources if not already downloaded
nltk.download('names', quiet=True)

# Feature extraction function
def extract_gender_features(name):
    name = name.lower()
    features = {
        "suffix": name[-1:],
        "suffix2": name[-2:] if len(name) > 1 else name[0],
        "suffix3": name[-3:] if len(name) > 2 else name[0],
        "suffix4": name[-4:] if len(name) > 3 else name[0],
        "suffix5": name[-5:] if len(name) > 4 else name[0],
        "suffix6": name[-6:] if len(name) > 5 else name[0],
        "prefix": name[:1],
        "prefix2": name[:2] if len(name) > 1 else name[0],
        "prefix3": name[:3] if len(name) > 2 else name[0],
        "prefix4": name[:4] if len(name) > 3 else name[0],
        "prefix5": name[:5] if len(name) > 4 else name[0]
    }
    return features

# Load the trained Naive Bayes classifier
bayes = load('gender_prediction.joblib')

# --- Custom CSS for Pro Dark Theme & Styling ---
custom_css = """
<style>
    /* Background & font */
    .main {
        background-color: #121212;
        color: #E0E0E0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Center container */
    .block-container {
        max-width: 600px;
        margin: auto;
        padding: 3rem 2rem 4rem 2rem;
        border-radius: 15px;
        background: #1e1e1e;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6);
    }

    /* Title styling */
    h1 {
        font-weight: 900 !important;
        font-size: 3.2rem !important;
        letter-spacing: 1.5px;
        color: #00bfa5;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    /* Subtitle styling */
    .subtitle {
        font-weight: 400;
        font-size: 1.2rem;
        color: #bbb;
        text-align: center;
        margin-bottom: 2.5rem;
        font-style: italic;
    }

    /* Input box */
    .stTextInput > div > div > input {
        background: #2b2b2b !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.8rem 1rem !important;
        font-size: 1.1rem !important;
        color: #e0e0e0 !important;
        box-shadow: inset 0 0 6px #00bfa5aa;
        transition: box-shadow 0.3s ease;
    }
    .stTextInput > div > div > input:focus {
        box-shadow: 0 0 8px #00bfa5ff !important;
        outline: none !important;
    }

    /* Button styling */
    div.stButton > button {
        background: linear-gradient(90deg, #00bfa5 0%, #00ffc8 100%);
        color: #121212;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 0.7rem 2.2rem;
        border-radius: 50px;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 191, 165, 0.7);
        cursor: pointer;
        transition: background 0.3s ease, transform 0.15s ease;
        display: block;
        margin: 1.5rem auto 3rem auto;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #00ffc8 0%, #00bfa5 100%);
        transform: scale(1.05);
    }

    /* Success message */
    .stAlertSuccess {
        border-radius: 10px;
        background-color: #004d40cc;
        color: #a7ffeb;
        font-weight: 600;
        font-size: 1.25rem;
        padding: 1rem 1.2rem;
        box-shadow: 0 0 12px #00bfa5cc;
        max-width: 100%;
        text-align: center;
        margin: auto;
    }

    /* Warning message */
    .stAlertWarning {
        border-radius: 10px;
        background-color: #bf3604cc;
        color: #ffd180;
        font-weight: 600;
        font-size: 1.15rem;
        padding: 1rem 1.2rem;
        max-width: 100%;
        text-align: center;
        margin: auto;
    }
</style>
"""

# Streamlit app
def main():
    st.markdown(custom_css, unsafe_allow_html=True)

    st.markdown("<h1>Gender Prediction App</h1>", unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Enter any name and let the AI guess the gender!</p>', unsafe_allow_html=True)

    input_name = st.text_input('Name', placeholder='Type a name here...')

    if st.button('Predict'):
        if input_name.strip():
            features = extract_gender_features(input_name)
            predicted_gender = bayes.classify(features)
            st.success(f'The predicted gender for "{input_name}" is: {predicted_gender}')
        else:
            st.warning('Please enter a name.')

if __name__ == '__main__':
    main()
