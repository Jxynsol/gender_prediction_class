import streamlit as st
import nltk
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
    /* Background with subtle pastel pattern */
    .main {
        background: #fff0f6 url("https://i.pinimg.com/originals/33/f3/d8/33f3d8eae6aebbf44692c9ae6a1bda4c.png") repeat;
        background-size: 150px 150px;
        font-family: 'Comic Sans MS', cursive, sans-serif;
        color: #d6336c;
        user-select: none;
        min-height: 100vh;
        padding-top: 2rem;
    }

    /* Container */
    .block-container {
        max-width: 600px;
        margin: 3rem auto 5rem auto;
        padding: 3rem 3rem 4rem 3rem;
        border-radius: 25px;
        background: #fff;
        box-shadow: 0 8px 32px rgba(214, 51, 108, 0.25);
        text-align: center;
        border: 4px solid #ff5c8d;
        position: relative;
    }

    /* Header image */
    .header-img {
        width: 120px;
        margin: 0 auto 1.8rem auto;
        display: block;
        animation: bounce 2.5s infinite ease-in-out;
    }

    /* Title */
    h1 {
        font-size: 3.8rem !important;
        font-weight: 900 !important;
        margin-bottom: 0.2rem;
        color: #ff5c8d;
        letter-spacing: 0.12em;
        text-shadow: 1px 1px 6px #ffbbcc;
    }

    /* Subtitle */
    .subtitle {
        font-style: italic;
        font-size: 1.4rem;
        margin-bottom: 2.5rem;
        color: #d6336c;
        font-weight: 600;
    }

    /* Input box */
    .stTextInput > div > div > input {
        background: #fff0f6 !important;
        border: 3px solid #ff5c8d !important;
        border-radius: 25px !important;
        padding: 1.2rem 1.5rem !important;
        font-size: 1.3rem !important;
        color: #d6336c !important;
        box-shadow: 0 5px 15px #ffb6c1;
        font-family: 'Comic Sans MS', cursive, sans-serif !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: #ff82ab !important;
        outline: none !important;
        box-shadow: 0 0 18px #ff82ab;
        transform: scale(1.02);
    }

    /* Button */
    div.stButton > button {
        background: linear-gradient(45deg, #ff5c8d, #ff82ab);
        color: white;
        font-weight: 800;
        font-size: 1.35rem;
        padding: 1.15rem 3.2rem;
        border-radius: 40px;
        border: none;
        box-shadow: 0 8px 20px #ff82abcc;
        cursor: pointer;
        transition: background 0.4s ease, transform 0.2s ease;
        font-family: 'Comic Sans MS', cursive, sans-serif;
        margin: 2.2rem auto 3rem auto;
        display: block;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    div.stButton > button:hover {
        background: linear-gradient(45deg, #ff82ab, #ff5c8d);
        transform: scale(1.08);
        box-shadow: 0 12px 30px #ff5c8dcc;
    }

    /* Success message */
    .stAlertSuccess {
        border-radius: 25px;
        background-color: #ffd6e8;
        color: #d6336c;
        font-weight: 700;
        font-size: 1.4rem;
        padding: 1.2rem 2.2rem;
        box-shadow: 0 0 28px #ff82abbb;
        max-width: 100%;
        margin: auto;
        text-align: center;
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }

    /* Warning message */
    .stAlertWarning {
        border-radius: 25px;
        background-color: #ffe3ec;
        color: #d6336c;
        font-weight: 600;
        font-size: 1.2rem;
        padding: 1.2rem 2.2rem;
        max-width: 100%;
        margin: auto;
        text-align: center;
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }

    /* Footer kitty icon */
    .footer-kitty {
        position: fixed;
        bottom: 1rem;
        right: 1rem;
        width: 100px;
        opacity: 0.7;
        animation: float 5s ease-in-out infinite;
        filter: drop-shadow(0 0 4px #ff5c8d);
        cursor: default;
        user-select: none;
        z-index: 9999;
    }

    /* Animations */
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-12px); }
    }
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-15px); }
    }
</style>
"""

def main():
    st.markdown(hello_kitty_css, unsafe_allow_html=True)

    # Hello Kitty header image from official or free source
    st.markdown("""
    <img class="header-img" src="https://upload.wikimedia.org/wikipedia/en/1/12/Hello_kitty.svg" alt="Hello Kitty" />
    """, unsafe_allow_html=True)

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

    # Footer Hello Kitty image
    st.markdown("""
    <img class="footer-kitty" src="https://upload.wikimedia.org/wikipedia/en/1/12/Hello_kitty.svg" alt="Hello Kitty" />
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
