import re

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

st.set_page_config(page_title="Hotel Review Sentiment", page_icon="🏨", layout="wide")


def clean(text):
    text = re.sub(r"http\S+|www\S+", "", str(text).lower())
    return re.sub(r"[^a-z\s]", " ", text)


@st.cache_data
def load_data():
    df = pd.read_csv("tripadvisor_hotel_reviews.csv")
    df["Clean"] = df["Review"].apply(clean)
    return df


@st.cache_resource
def train_model(df):
    data = df[df.Rating.isin([1, 5])]  # extreme ratings only, like the notebook
    y = (data.Rating == 5).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(data.Clean, y, test_size=0.2, random_state=42)
    vec = CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    Xtr = vec.fit_transform(X_train)  # fit on training data only
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=30, random_state=42)
    clf.fit(Xtr, y_train)
    pred = clf.predict(vec.transform(X_test))
    return vec, clf, accuracy_score(y_test, pred), f1_score(y_test, pred, average="macro"), \
        confusion_matrix(y_test, pred)


def top_words(texts, n=15):
    vec = CountVectorizer(stop_words="english", max_features=n)
    counts = vec.fit_transform(texts).sum(axis=0).A1
    return pd.DataFrame({"Word": vec.get_feature_names_out(), "Count": counts}).sort_values("Count")


df = load_data()

st.title("🏨 Hotel Review Sentiment")
st.caption("A neural network that reads a hotel review and tells you if it is positive or negative.")

with st.spinner("Training the model (first run only)..."):
    vec, clf, acc, f1, cm = train_model(df)

tab1, tab2, tab3 = st.tabs(["Try it", "Data insights", "Model"])

with tab1:
    text = st.text_area("Write a hotel review (in English)", height=150,
                        placeholder="The room was clean and the staff were super friendly...")
    if st.button("Analyze", type="primary") and text.strip():
        pos = clf.predict_proba(vec.transform([clean(text)]))[0][1]
        label = "Positive 😊" if pos >= 0.5 else "Negative 😞"
        st.metric("Sentiment", label, f"{max(pos, 1 - pos) * 100:.1f}% confidence")
        st.progress(float(pos), text=f"Positive probability: {pos:.2f}")

with tab2:
    col1, col2 = st.columns(2)
    ratings = df.Rating.value_counts().sort_index().rename_axis("Rating").reset_index(name="Reviews")
    col1.plotly_chart(px.bar(ratings, x="Rating", y="Reviews", title="Reviews per rating"),
                      use_container_width=True)
    df["Length"] = df.Clean.str.split().str.len()
    col2.plotly_chart(px.box(df, x="Rating", y="Length", title="Review length (words) per rating"),
                      use_container_width=True)
    col1.plotly_chart(px.bar(top_words(df[df.Rating == 1].Clean), x="Count", y="Word", orientation="h",
                             title="Top words in 1-star reviews"), use_container_width=True)
    col2.plotly_chart(px.bar(top_words(df[df.Rating == 5].Clean), x="Count", y="Word", orientation="h",
                             title="Top words in 5-star reviews"), use_container_width=True)

with tab3:
    m1, m2 = st.columns(2)
    m1.metric("Test accuracy", f"{acc:.3f}")
    m2.metric("Macro F1", f"{f1:.3f}")
    st.plotly_chart(px.imshow(cm, text_auto=True, x=["Negative", "Positive"], y=["Negative", "Positive"],
                              labels=dict(x="Predicted", y="Actual"), color_continuous_scale="Blues",
                              title="Confusion matrix"), use_container_width=True)
    st.caption("Trained on 1-star (negative) and 5-star (positive) reviews only.")