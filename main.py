import streamlit as st
import nltk
from nltk import NaiveBayesClassifier
from joblib import load

nltk.download('names', quiet=True)

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

bayes = load('gender_prediction.joblib')

cyberpunk_css = """
<style>
    /* Background */
    .main {
        background: radial-gradient(circle at top left, #0f0c29, #302b63, #24243e);
        color: #f0f0f0;
        font-family: 'Orbitron', 'Courier New', Courier, monospace;
        user-select: none;
    }

    /* Container */
    .block-container {
        max-width: 600px;
        margin: 3rem auto;
        padding: 2.5rem 3rem 3rem 3rem;
        border-radius: 15px;
        background: rgba(10, 10, 30, 0.85);
        box-shadow:
            0 0 15px #ff0080,
            0 0 30px #00fff7,
            0 0 45px #ff0080,
            0 0 60px #00fff7;
        border: 2px solid #ff0080;
        text-align: center;
    }

    /* Title */
    h1 {
        font-weight: 900 !important;
        font-size: 3.5rem !important;
        letter-spacing: 0.15em;
        color: #ff0080;
        text-shadow:
            0 0 10px #ff0080,
            0 0 20px #ff0080,
            0 0 30px #ff0080;
        margin-bottom: 0.3rem;
        font-family: 'Orbitron', monospace;
    }

    /* Subtitle */
    .subtitle {
        font-style: italic;
        font-weight: 600;
        font-size: 1.2rem;
        color: #00fff7;
        text-shadow: 0 0 5px #00fff7;
        margin-bottom: 3rem;
    }

    /* Input box */
    .stTextInput > div > div > input {
        background: #1a1a40 !important;
        border: 2px solid #ff0080 !important;
        border-radius: 8px !important;
        padding: 1rem 1.2rem !important;
        font-size: 1.2rem !important;
        color: #00fff7 !important;
        box-shadow:
            inset 0 0 6px #ff0080,
            0 0 10px #00fff7;
        font-family: 'Orbitron', monospace !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: #00fff7 !important;
        box-shadow:
            inset 0 0 10px #00fff7,
            0 0 20px #ff0080;
        outline: none !important;
    }

    /* Button */
    div.stButton > button {
        background: linear-gradient(90deg, #ff0080, #00fff7);
        color: #121212;
        font-weight: 900;
        font-size: 1.3rem;
        padding: 1rem 2.5rem;
        border-radius: 50px;
        border: none;
        box-shadow:
            0 0 10px #ff0080,
            0 0 20px #00fff7;
        cursor: pointer;
        transition: filter 0.3s ease, transform 0.15s ease;
        font-family: 'Orbitron', monospace;
        margin: 1.8rem auto 3rem auto;
        display: block;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    div.stButton > button:hover {
        filter: brightness(1.3);
        transform: scale(1.07);
    }

    /* Success Message */
    .stAlertSuccess {
        border-radius: 12px;
        background-color: #00fff722;
        color: #00fff7;
        font-weight: 700;
        font-size: 1.3rem;
        padding: 1rem 1.5rem;
        box-shadow:
            0 0 12px #00fff7,
            0 0 25px #00fff7cc;
        max-width: 100%;
        margin: auto;
        font-family: 'Orbitron', monospace;
        text-align: center;
        letter-spacing: 0.05em;
    }

    /* Warning Message */
    .stAlertWarning {
        border-radius: 12px;
        background-color: #ff008022;
        color: #ff0080;
        font-weight: 700;
        font-size: 1.25rem;
        padding: 1rem 1.5rem;
        max-width: 100%;
        margin: auto;
        text-align: center;
        font-family: 'Orbitron', monospace;
        letter-spacing: 0.05em;
    }
</style>

<!-- Import Orbitron font -->
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&display=swap" rel="stylesheet">
"""

def main():
    st.markdown(cyberpunk_css, unsafe_allow_html=True)

    st.markdown("<h1>GENDER PREDICTION</h1>", unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Enter a name, and the AI will reveal its gender.</p>', unsafe_allow_html=True)

    name = st.text_input('Name', placeholder='e.g. Neo, Trinity...')

    if st.button('Predict'):
        if name.strip():
            features = extract_gender_features(name)
            gender = bayes.classify(features)
            st.success(f'⚡ The predicted gender for "{name}" is: {gender} ⚡')
        else:
            st.warning('⚠️ Please enter a name to proceed.')

if __name__ == "__main__":
    main()
