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

cyberpunk_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap');
    @keyframes neonPulse {
        0%, 100% {
            text-shadow:
                0 0 5px #ff3c78,
                0 0 10px #ff3c78,
                0 0 20px #ff3c78,
                0 0 40px #ff3c78,
                0 0 80px #ff3c78;
        }
        50% {
            text-shadow:
                0 0 10px #ff6bbd,
                0 0 20px #ff6bbd,
                0 0 40px #ff6bbd,
                0 0 60px #ff6bbd,
                0 0 100px #ff6bbd;
        }
    }
    body {
        background: #0d001f;
        color: #ff3c78;
        font-family: 'Orbitron', monospace, monospace;
        margin: 0;
        padding: 0;
        min-height: 100vh;
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        padding: 2rem 1rem;
    }
    .card {
        background: #12002b;
        border: 3px solid #ff3c78;
        border-radius: 20px;
        padding: 3rem 3.5rem;
        max-width: 500px;
        width: 90vw;
        box-shadow:
            0 0 15px #ff3c78,
            0 0 30px #ff6bbd,
            0 0 40px #ff3c78;
        text-align: center;
        position: relative;
        z-index: 1;
    }
    h1 {
        font-weight: 900;
        font-size: 2.8rem;
        letter-spacing: 0.25em;
        text-transform: uppercase;
        animation: neonPulse 3s ease-in-out infinite;
        margin-bottom: 0.3rem;
    }
    .subtitle {
        font-size: 1.1rem;
        font-weight: 600;
        font-style: italic;
        color: #ff6bbd;
        margin-bottom: 2rem;
        letter-spacing: 0.1em;
    }
    input[type="text"] {
        background: #230040;
        border: 2px solid #ff3c78;
        border-radius: 15px;
        padding: 1rem 1.5rem;
        width: 100%;
        font-size: 1.3rem;
        color: #ff6bbd;
        font-family: 'Orbitron', monospace, monospace;
        box-shadow:
            0 0 10px #ff3c78,
            inset 0 0 15px #ff6bbd;
        transition: 0.3s ease;
        outline: none;
        letter-spacing: 0.1em;
    }
    input[type="text"]:focus {
        border-color: #ff6bbd;
        box-shadow:
            0 0 20px #ff6bbd,
            inset 0 0 25px #ff3c78;
        transform: scale(1.03);
    }
    button {
        margin-top: 1.5rem;
        background: linear-gradient(90deg, #ff3c78, #ff6bbd);
        border: none;
        border-radius: 50px;
        color: white;
        font-family: 'Orbitron', monospace, monospace;
        font-weight: 900;
        font-size: 1.4rem;
        padding: 1rem 3rem;
        cursor: pointer;
        box-shadow:
            0 0 20px #ff3c78,
            0 0 30px #ff6bbd;
        transition: 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        user-select: none;
    }
    button:hover {
        background: linear-gradient(90deg, #ff6bbd, #ff3c78);
        box-shadow:
            0 0 40px #ff6bbd,
            0 0 50px #ff3c78,
            0 0 60px #ff6bbd;
        transform: scale(1.1);
    }
    .stAlertSuccess, .stAlertWarning {
        font-family: 'Orbitron', monospace, monospace;
        font-weight: 700;
        font-size: 1.2rem;
        border-radius: 15px;
        padding: 1rem 2rem;
        margin-top: 1.8rem;
        text-align: center;
        max-width: 480px;
        margin-left: auto;
        margin-right: auto;
    }
    .stAlertSuccess {
        background: #33002f;
        color: #ff6bbd;
        box-shadow: 0 0 30px #ff3c78;
    }
    .stAlertWarning {
        background: #330022;
        color: #ff3c78;
        box-shadow: 0 0 30px #ff3c78;
    }
</style>
"""

def main():
    st.markdown(cyberpunk_css, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
            <h1>Gender Predictor</h1>
            <p class="subtitle">TYPE A NAME — GET YOUR KAWKAW GENDER! ⚡</p>
        </div>
        """, unsafe_allow_html=True
    )

    name = st.text_input("NAME", "")

    if st.button("PREDICT"):
        if name.strip() == "":
            st.warning("🚨 Please enter a name, cyber warrior!")
        else:
            features = extract_gender_features(name)
            gender = bayes.classify(features)
            st.success(f"✨ The predicted gender for \"{name}\" is: {gender} ✨")

if __name__ == "__main__":
    main()
