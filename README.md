# ============================================================
# AI-BASED FAKE NEWS DETECTION SYSTEM
# Minor Subject Capstone Project - AIMLP-004
# Artificial Intelligence and its Applications
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import os

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Fake News Detection",
    page_icon="📰",
    layout="wide"
)


# ============================================================
# 2. CUSTOM DESIGN
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #f4f7fb;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #1e3a8a;
}

.sub-title {
    text-align: center;
    font-size: 18px;
    color: #555;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. TEXT PREPROCESSING
# ============================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Convert text to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# 4. LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    if not os.path.exists("Fake.csv") or not os.path.exists("True.csv"):

        return None

    fake_data = pd.read_csv("Fake.csv")

    real_data = pd.read_csv("True.csv")

    # Assign labels
    fake_data["label"] = 0
    real_data["label"] = 1

    # Combine datasets
    data = pd.concat(
        [fake_data, real_data],
        ignore_index=True
    )

    # Combine title and article
    data["content"] = (
        data["title"].fillna("") + " " +
        data["text"].fillna("")
    )

    # Remove duplicates
    data = data.drop_duplicates(subset=["content"])

    # Clean text
    data["content"] = data["content"].apply(clean_text)

    # Remove empty text
    data = data[data["content"].str.len() > 0]

    return data


# ============================================================
# 5. TRAIN MACHINE LEARNING MODELS
# ============================================================

@st.cache_resource
def train_models(data):

    X = data["content"]

    y = data["label"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )


    # --------------------------------------------------------
    # LOGISTIC REGRESSION PIPELINE
    # --------------------------------------------------------

    logistic_model = Pipeline([

        (
            "tfidf",

            TfidfVectorizer(

                max_features=50000,

                ngram_range=(1, 2),

                stop_words="english",

                min_df=2,

                max_df=0.95
            )
        ),

        (
            "classifier",

            LogisticRegression(

                max_iter=1000,

                class_weight="balanced"
            )
        )

    ])


    # --------------------------------------------------------
    # NAIVE BAYES PIPELINE
    # --------------------------------------------------------

    naive_bayes_model = Pipeline([

        (
            "tfidf",

            TfidfVectorizer(

                max_features=50000,

                ngram_range=(1, 2),

                stop_words="english",

                min_df=2,

                max_df=0.95
            )
        ),

        (
            "classifier",

            MultinomialNB()
        )

    ])


    # --------------------------------------------------------
    # TRAIN MODELS
    # --------------------------------------------------------

    logistic_model.fit(X_train, y_train)

    naive_bayes_model.fit(X_train, y_train)


    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    logistic_predictions = logistic_model.predict(X_test)

    nb_predictions = naive_bayes_model.predict(X_test)


    # --------------------------------------------------------
    # MODEL EVALUATION
    # --------------------------------------------------------

    results = {}

    results["Logistic Regression"] = {

        "accuracy": accuracy_score(
            y_test,
            logistic_predictions
        ),

        "precision": precision_score(
            y_test,
            logistic_predictions
        ),

        "recall": recall_score(
            y_test,
            logistic_predictions
        ),

        "f1": f1_score(
            y_test,
            logistic_predictions
        ),

        "confusion_matrix": confusion_matrix(
            y_test,
            logistic_predictions
        ),

        "report": classification_report(
            y_test,
            logistic_predictions,
            target_names=["Fake News", "Real News"],
            output_dict=True
        )
    }


    results["Naive Bayes"] = {

        "accuracy": accuracy_score(
            y_test,
            nb_predictions
        ),

        "precision": precision_score(
            y_test,
            nb_predictions
        ),

        "recall": recall_score(
            y_test,
            nb_predictions
        ),

        "f1": f1_score(
            y_test,
            nb_predictions
        ),

        "confusion_matrix": confusion_matrix(
            y_test,
            nb_predictions
        ),

        "report": classification_report(
            y_test,
            nb_predictions,
            target_names=["Fake News", "Real News"],
            output_dict=True
        )
    }


    return logistic_model, naive_bayes_model, results, X_test, y_test


# ============================================================
# 6. HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📰 AI-Based Fake News Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Artificial Intelligence | NLP | Machine Learning</div>',
    unsafe_allow_html=True
)

st.markdown("""
This application uses Machine Learning and Natural Language Processing
to analyse news content and classify it based on patterns learned
from labelled training data.
""")

st.divider()


# ============================================================
# 7. SIDEBAR
# ============================================================

st.sidebar.title("Project Information")

st.sidebar.markdown("""
**Project:** AI Fake News Detection

**Domain:** Artificial Intelligence

**Algorithms:**
- Logistic Regression
- Naive Bayes

**Feature Extraction:** TF-IDF

**Language:** Python

**Framework:** Streamlit
""")

st.sidebar.divider()

st.sidebar.markdown("""
### Project Workflow

1. Dataset loading
2. Text preprocessing
3. TF-IDF vectorization
4. Model training
5. Model evaluation
6. News prediction
""")

st.sidebar.warning(
    "Predictions are estimates and are not independent fact-checks."
)


# ============================================================
# 8. DATASET LOADING
# ============================================================

data = load_dataset()


if data is None:

    st.error("Dataset files were not found.")

    st.markdown("""
    ### How to fix this

    Download the Fake and Real News Dataset.

    Place the following files in the same folder as this Python file:

    - `Fake.csv`
    - `True.csv`

    Then restart the application.
    """)

    st.stop()


# ============================================================
# 9. DATASET INFORMATION
# ============================================================

st.subheader("📊 Dataset Information")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total News Articles",
    f"{len(data):,}"
)

col2.metric(
    "Fake News Articles",
    f"{(data['label'] == 0).sum():,}"
)

col3.metric(
    "Real News Articles",
    f"{(data['label'] == 1).sum():,}"
)


# ============================================================
# 10. TRAIN MODELS
# ============================================================

with st.spinner("Training machine learning models..."):

    logistic_model, nb_model, results, X_test, y_test = train_models(data)


st.success("Machine learning models trained successfully!")


# ============================================================
# 11. MODEL COMPARISON
# ============================================================

st.subheader("🤖 Machine Learning Model Comparison")

comparison_df = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Naive Bayes"
    ],

    "Accuracy": [
        results["Logistic Regression"]["accuracy"],
        results["Naive Bayes"]["accuracy"]
    ],

    "Precision": [
        results["Logistic Regression"]["precision"],
        results["Naive Bayes"]["precision"]
    ],

    "Recall": [
        results["Logistic Regression"]["recall"],
        results["Naive Bayes"]["recall"]
    ],

    "F1 Score": [
        results["Logistic Regression"]["f1"],
        results["Naive Bayes"]["f1"]
    ]

})

st.dataframe(
    comparison_df.style.format({
        "Accuracy": "{:.2%}",
        "Precision": "{:.2%}",
        "Recall": "{:.2%}",
        "F1 Score": "{:.2%}"
    }),
    use_container_width=True
)


# ============================================================
# 12. SELECT MODEL
# ============================================================

st.subheader("Select Machine Learning Model")

selected_model_name = st.selectbox(

    "Choose the model for news detection:",

    [
        "Logistic Regression",
        "Naive Bayes"
    ]
)


if selected_model_name == "Logistic Regression":

    selected_model = logistic_model

else:

    selected_model = nb_model


selected_results = results[selected_model_name]


# ============================================================
# 13. DISPLAY PERFORMANCE
# ============================================================

st.subheader("📈 Selected Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Accuracy",
    f"{selected_results['accuracy']:.2%}"
)

col2.metric(
    "Precision",
    f"{selected_results['precision']:.2%}"
)

col3.metric(
    "Recall",
    f"{selected_results['recall']:.2%}"
)

col4.metric(
    "F1 Score",
    f"{selected_results['f1']:.2%}"
)


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

st.subheader("Confusion Matrix")

cm = selected_results["confusion_matrix"]

fig, ax = plt.subplots(figsize=(6, 4))

im = ax.imshow(cm)

ax.set_xticks([0, 1])

ax.set_yticks([0, 1])

ax.set_xticklabels(["Fake News", "Real News"])

ax.set_yticklabels(["Fake News", "Real News"])

ax.set_xlabel("Predicted Label")

ax.set_ylabel("Actual Label")

ax.set_title("Confusion Matrix")

for i in range(2):

    for j in range(2):

        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=14
        )

fig.colorbar(im)

st.pyplot(fig)

plt.close(fig)


# ============================================================
# 15. NEWS PREDICTION SECTION
# ============================================================

st.divider()

st.subheader("🔍 Fake News Detection")

news_input = st.text_area(

    "Enter news headline or article:",

    height=250,

    placeholder="Paste news content here..."
)


# ============================================================
# 16. PREDICT BUTTON
# ============================================================

if st.button(
    "🚀 Analyse News",
    use_container_width=True
):

    if len(news_input.strip()) < 30:

        st.warning(
            "Please enter a news article containing "
            "at least 30 characters."
        )

    else:

        # Make prediction
        prediction = selected_model.predict(
            [news_input]
        )[0]

        # Calculate probabilities
        probabilities = selected_model.predict_proba(
            [news_input]
        )[0]

        # Confidence
        confidence = max(probabilities) * 100

        # Text statistics
        words = re.findall(
            r"\b\w+\b",
            news_input
        )

        word_count = len(words)

        character_count = len(news_input)


        # ----------------------------------------------------
        # PREDICTION RESULT
        # ----------------------------------------------------

        st.subheader("Prediction Result")


        if prediction == 0:

            st.error("🚨 POTENTIALLY FAKE NEWS")

            st.write("""
            The text resembles patterns associated with Fake News
            in the training dataset.
            """)

        else:

            st.success("✅ POTENTIALLY REAL NEWS")

            st.write("""
            The text resembles patterns associated with Real News
            in the training dataset.
            """)


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )


        # ----------------------------------------------------
        # TEXT STATISTICS
        # ----------------------------------------------------

        st.subheader("Text Analysis")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Word Count",
            word_count
        )

        col2.metric(
            "Characters",
            character_count
        )

        col3.metric(
            "Selected Model",
            selected_model_name
        )


        # ----------------------------------------------------
        # PREDICTION PROBABILITY
        # ----------------------------------------------------

        st.subheader("Prediction Probability")

        fake_probability = probabilities[0]

        real_probability = probabilities[1]


        st.write(
            f"Fake News: {fake_probability:.2%}"
        )

        st.progress(
            float(fake_probability)
        )


        st.write(
            f"Real News: {real_probability:.2%}"
        )

        st.progress(
            float(real_probability)
        )


        # ----------------------------------------------------
        # DISCLAIMER
        # ----------------------------------------------------

        st.warning("""
        **Important Disclaimer**

        This application uses Machine Learning to analyse text
        patterns. It cannot independently verify news claims.

        A prediction may be incorrect. Please consult trusted
        news sources and independent fact-checking services
        before deciding whether a claim is true or false.
        """)


# ============================================================
# 17. CLASSIFICATION REPORT
# ============================================================

st.divider()

st.subheader("📋 Detailed Classification Report")

report_df = pd.DataFrame(
    selected_results["report"]
).transpose()

st.dataframe(
    report_df.round(3),
    use_container_width=True
)


# ============================================================
# 18. FOOTER
# ============================================================

st.divider()

st.markdown("""
<div style="text-align:center">

### AI-Based Fake News Detection System

**Minor Subject Capstone Project**

**Subject Code:** AIMLP-004

**Domain:** Artificial Intelligence and its Applications

Developed using Python, NLP and Machine Learning

</div>
""", unsafe_allow_html=True)
