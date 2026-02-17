# 🧠 Emotion Detection System using NLP

This project is a **Natural Language Processing (NLP) based Emotion Detection System** that predicts the emotional state expressed in a given text.  
It uses **TF-IDF feature extraction** and **Logistic Regression** for multi-class emotion classification, along with a **Streamlit-based user interface** for real-time predictions.

---

## 📌 Features

- Predicts emotions from text input
- Supports **6 emotion classes**:
  - Anger
  - Fear
  - Joy
  - Love
  - Sadness
  - Surprise
- Displays **prediction confidence**
- Handles low-confidence and ambiguous inputs gracefully
- Clean, professional UI built with Streamlit
- Model persistence using `joblib`

---

## 🛠️ Tech Stack

- **Python**
- **Scikit-learn**
- **TF-IDF Vectorizer**
- **Logistic Regression**
- **Streamlit**
- **Pandas & NumPy**
- **Joblib**

---

## 📂 Project Structure

.
├── app.py # Streamlit UI application
├── train_model.py # Model training script
├── Chat.ipynb # Notebook for training & analysis
├── model.pkl # Trained Logistic Regression model
├── vectorizer.pkl # Trained TF-IDF vectorizer
├── data/
│ └── train.txt # Emotion-labeled text dataset
└── README.md


---

## 📊 Dataset Description

- The dataset consists of text sentences labeled with emotions.
- Format:


---

## ⚙️ Model Details

- **Feature Extraction**: TF-IDF (Term Frequency–Inverse Document Frequency)
- **Classifier**: Logistic Regression
- **Training Strategy**:
- Multi-class classification
- Class balancing using `class_weight="balanced"`
- **Model Output**:
- Predicted emotion label
- Confidence score

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install pandas scikit-learn streamlit joblib


License

This project is for educational purposes only.


