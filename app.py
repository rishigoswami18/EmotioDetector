import streamlit as st
import joblib

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Emotion Detection System",
    page_icon="🧠",
    layout="centered"
)

# --------------------------------------------------
# Load trained components
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# --------------------------------------------------
# App header
# --------------------------------------------------
st.title("🧠 Emotion Detection System")
st.caption(
    "A Natural Language Processing application using "
    "TF-IDF and Logistic Regression"
)

st.divider()

# --------------------------------------------------
# Input section
# --------------------------------------------------
st.subheader("🔎 Input Text")

user_text = st.text_area(
    label="Enter a sentence or paragraph:",
    placeholder="Example: I feel lonely and overwhelmed today...",
    height=150
)

# --------------------------------------------------
# Prediction logic
# --------------------------------------------------
if st.button("Analyze Emotion", use_container_width=True):

    if user_text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Vectorize input
        text_vector = vectorizer.transform([user_text])

        # Predict
        prediction = model.predict(text_vector)[0]
        probabilities = model.predict_proba(text_vector)[0]

        # Confidence for predicted class
        class_index = list(model.classes_).index(prediction)
        confidence = probabilities[class_index]

        st.divider()

        # --------------------------------------------------
        # Output section
        # --------------------------------------------------
        st.subheader("📊 Prediction Result")

        st.success(f"**Predicted Emotion:** {prediction.upper()}")
        st.metric(
            label="Model Confidence",
            value=f"{confidence:.2f}"
        )

        # --------------------------------------------------
        # Confidence interpretation
        # --------------------------------------------------
        if confidence < 0.50:
            st.warning(
                "⚠️ The model is **not confident** about this prediction. "
                "The text may contain mixed or ambiguous emotions."
            )
        elif confidence < 0.70:
            st.info(
                "ℹ️ The model is **moderately confident**. "
                "Some emotional overlap may exist."
            )
        else:
            st.success(
                "✅ The model is **highly confident** in this prediction."
            )

        # --------------------------------------------------
        # Explanation box (professional touch)
        # --------------------------------------------------
        with st.expander("ℹ️ How to interpret this result"):
            st.write(
                """
                - This system uses **TF-IDF** to convert text into numerical features.
                - A **Logistic Regression** classifier predicts one of six emotions:
                  **anger, fear, joy, love, sadness, surprise**.
                - Confidence reflects how strongly the model supports the predicted emotion.
                - Lower confidence usually indicates **mixed or subtle emotional content**.
                """
            )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()
st.caption(
    "Developed as an NLP project using classical machine learning techniques."
)
