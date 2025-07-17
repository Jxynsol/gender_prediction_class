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

hello_kitty_css = """
<style>
    /* Background & font */
    .main {
        background: #fff0f6;
        font-family: 'Comic Sans MS', cursive, sans-serif;
        color: #d6336c;
        user-select: none;
    }

    /* Container */
    .block-container {
        max-width: 550px;
        margin: 3rem auto;
        padding: 3rem 3rem 4rem 3rem;
        border-radius: 25px;
        background: #fff;
        box-shadow: 0 8px 24px rgba(214, 51, 108, 0.3);
        text-align: center;
        border: 3px solid #ff5c8d;
    }

    /* Title */
    h1 {
        font-size: 3.5rem !important;
        font-weight: 900 !important;
        margin-bottom: 0.2rem;
        color: #ff5c8d;
        letter-spacing: 0.1em;
        text-shadow: 1px 1px 4px #ffbbcc;
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }

    /* Subtitle */
    .subtitle {
        font-style: italic;
        font-size: 1.3rem;
        margin-bottom: 2.5rem;
        color: #d6336c;
        font-weight: 600;
    }

    /* Input box */
    .stTextInput > div > div > input {
        background: #fff0f6 !important;
        border: 2px solid #ff5c8d !important;
        border-radius: 20px !important;
        padding: 1rem 1.2rem !important;
        font-size: 1.2rem !important;
        color: #d6336c !important;
        box-shadow: 0 4px 8px #ffb6c1;
        font-family: 'Comic Sans MS', cursive, sans-serif !important;
        transition: border-color 0.3s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: #ff82ab !important;
        outline: none !important;
        box-shadow: 0 0 10px #ff82ab;
    }

    /* Button */
    div.stButton > button {
        background: #ff5c8d;
        color: white;
        font-weight: 700;
        font-size: 1.3rem;
        padding: 1rem 3rem;
        border-radius: 30px;
        border: none;
        box-shadow: 0 6px 15px #ff82ab;
        cursor: pointer;
        transition: background 0.3s ease, transform 0.15s ease;
        font-family: 'Comic Sans MS', cursive, sans-serif;
        margin: 2rem auto 3rem auto;
        display: block;
    }
    div.stButton > button:hover {
        background: #ff82ab;
        transform: scale(1.05);
    }

    /* Success message */
    .stAlertSuccess {
        border-radius: 20px;
        background-color: #ffd6e8;
        color: #d6336c;
        font-weight: 700;
        font-size: 1.3rem;
        padding: 1rem 2rem;
        box-shadow: 0 0 20px #ff82ab99;
        max-width: 100%;
        margin: auto;
        text-align: center;
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }

    /* Warning message */
    .stAlertWarning {
        border-radius: 20px;
        background-color: #ffe3ec;
        color: #d6336c;
        font-weight: 600;
        font-size: 1.15rem;
        padding: 1rem 2rem;
        max-width: 100%;
        margin: auto;
        text-align: center;
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }

    /* Hello Kitty icon (simple svg) */
    .hello-kitty-icon {
        margin-bottom: 1rem;
        width: 80px;
        height: 80px;
        filter: drop-shadow(0 0 2px #ff5c8d);
    }
</style>
"""

hello_kitty_svg = """
<svg class="hello-kitty-icon" viewBox="0 0 512 512" fill="#ff5c8d" xmlns="http://www.w3.org/2000/svg">
  <path d="M256 48c-73 0-134 59-134 132s61 132 134 132 134-59 134-132-61-132-134-132zm0 240c-59 0-106-48-106-108s47-108 106-108 106 48 106 108-47 108-106 108z"/>
  <circle cx="170" cy="220" r="15"/>
  <circle cx="342" cy="220" r="15"/>
  <path d="M256 286c-22 0-40 18-40 40h80c0-22-18-40-40-40z"/>
</svg>
"""

def main():
    st.markdown(hello_kitty_css, unsafe_allow_html=True)
    st.markdown(hello_kitty_svg, unsafe_allow_html=True)

    st.markdown('<h1>Hello Kitty Gender Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Type a name and the magic will tell you the gender! 🎀</p>', unsafe_allow_html=True)

    name = st.text_input('Name', placeholder='e.g. Mimi, Taro...')

    if st.button('Predict'):
        if name.strip():
            features = extract_gender_features(name)
            gender = bayes.classify(features)
            st.success(f'🎉 The predicted gender for "{name}" is: {gender} 🎉')
        else:
            st.warning('Please enter a name, nya!')

if __name__ == '__main__':
    main()
