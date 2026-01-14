import streamlit as st
from nlp_pipeline import train_model, predict_message

# =========================
# Load Model ONCE
# =========================
@st.cache_resource
def load_model():
    return train_model("spam.csv")

model, vectorizer = load_model()

# =========================
# UI
# =========================
st.set_page_config(page_title="Fraud & Phishing Detector", layout="centered")

st.title("📨 Fraud & Phishing Risk Detection System")
st.write("Industry-grade NLP system with risk scoring & explainability.")

user_input = st.text_area("Enter a message to analyze")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        result = predict_message(user_input, model, vectorizer)

        st.subheader("Result")
        st.write(f"**Risk Level:** {result['risk']}")
        st.write(f"**Spam Probability:** {result['spam_probability']}")

        with st.expander("Explanation"):
            st.write("Message was analyzed using NLP patterns, phrase context, and statistical probabilities.")
            st.write("Cleaned text used by model:")
            st.code(result['cleaned_text'])
