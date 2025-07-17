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

# Load your pre-trained Naive Bayes classifier here
bayes = load('gender_prediction.joblib')

# Kawaii pink CSS styling with Hello Kitty image fix
kawaii_css = """
<style>
    .main {
        background: #fff0f6;
        font-family: 'Comic Sans MS', cursive, sans-serif;
        color: #4a2c40;
        min-height: 100vh;
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem 1rem;
    }
    .card {
        background: white;
        border-radius: 24px;
        padding: 3rem 3rem 4rem 3rem;
        box-shadow: 0 6px 25px rgba(255, 92, 141, 0.3);
        max-width: 600px;
        width: 100%;
        border: 3px solid #ff5c8d;
        text-align: center;
    }
    .card img {
        width: 100px;
        margin-bottom: 1.5rem;
    }
    h1 {
        font-weight: 900;
        font-size: 3rem;
        letter-spacing: 0.1em;
        color: #ff5c8d;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 6px #ffbbcc;
    }
    .subtitle {
        font-style: italic;
        color: #d6336c;
        font-weight: 600;
        margin-bottom: 2rem;
        font-size: 1.2rem;
    }
    input[type="text"] {
        width: 100%;
        padding: 1rem 1.3rem;
        font-size: 1.2rem;
        border-radius: 20px;
        border: 2px solid #ff5c8d;
        box-shadow: 0 5px 15px #ffb6c1;
        font-family: 'Comic Sans MS', cursive, sans-serif;
        outline: none;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 2rem;
        color: #d6336c;
        background: #fff0f6;
    }
    input[type="text"]:focus {
        border-color: #ff82ab;
        box-shadow: 0 0 18px #ff82ab;
        transform: scale(1.02);
    }
    button {
        background: linear-gradient(45deg, #ff5c8d, #ff82ab);
        color: white;
        font-weight: 700;
        font-size: 1.3rem;
        padding: 1rem 3rem;
        border-radius: 40px;
        border: none;
        box-shadow: 0 8px 20px #ff82abcc;
        cursor: pointer;
        transition: background 0.4s ease, transform 0.2s ease;
        font-family: 'Comic Sans MS', cursive, sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    button:hover {
        background: linear-gradient(45deg, #ff82ab, #ff5c8d);
        transform: scale(1.05);
        box-shadow: 0 12px 30px #ff5c8dcc;
    }
    .stAlertSuccess {
        border-radius: 20px;
        background-color: #ffd6e8;
        color: #d6336c;
        font-weight: 700;
        font-size: 1.3rem;
        padding: 1rem 2rem;
        box-shadow: 0 0 28px #ff82abbb;
        margin-top: 1.5rem;
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }
    .stAlertWarning {
        border-radius: 20px;
        background-color: #ffe3ec;
        color: #d6336c;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 1rem 2rem;
        margin-top: 1.5rem;
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }
</style>
"""

def main():
    st.markdown(kawaii_css, unsafe_allow_html=True)

    st.markdown("""
        <div class="card">
            <img src="https://upload.wikimedia.org/wikipedia/en/1/12/Hello_kitty.svg" alt="Hello Kitty" />
            <h1>Hello Kitty Gender Predictor</h1>
            <p class="subtitle">Type a name and the magic will tell you the gender! 🎀</p>
        </div>
    """, unsafe_allow_html=True)

    name = st.text_input("Name", "")

    if st.button("Predict"):
        if name.strip() == "":
            st.warning("Please enter a name, nya!")
        else:
            features = extract_gender_features(name)
            gender = bayes.classify(features)
            st.success(f"🎉 The predicted gender for \"{name}\" is: {gender} 🎉")

if __name__ == "__main__":
    main()
