import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

tfidf = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))
models = {
    "Logistic Regression": pickle.load(open("model/log_reg.pkl", "rb")),
    "Naive Bayes": pickle.load(open("model/naive_bayes.pkl", "rb")),
    "Random Forest": pickle.load(open("model/random_forest.pkl", "rb")),
}

st.set_page_config(page_title="🎬 Sentiment Analysis", layout="wide")

with open("frontend/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>🎬 Movie Review Sentiment Analysis</h1>", unsafe_allow_html=True)
st.write("Enter a movie review and find out whether it’s **positive** or **negative**!")

st.sidebar.header("⚙️ Settings")
selected_model = st.sidebar.selectbox("Choose Model", list(models.keys()))

user_input = st.text_area("📝 Enter your review here:")
if st.button("Analyze Sentiment"):
    if user_input.strip():
        vectorized = tfidf.transform([user_input])
        pred = models[selected_model].predict(vectorized)[0]
        sentiment = "Positive 😀" if pred == 1 else "Negative 😡"
        st.success(f"Model Used: {selected_model} → Prediction: **{sentiment}**")
    else:
        st.warning("Please enter a review.")

st.subheader("📊 Dataset Insights")

url = "https://raw.githubusercontent.com/Shaik-Saaduddin/Sentiment-Analysis-on-Movie-Reviews/main/IMDB_Dataset.csv"
df = pd.read_csv(url)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

col1, col2 = st.columns(2)

with col1:
    st.write("### Sentiment Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=df['sentiment'], palette="coolwarm", ax=ax)
    ax.set_xticklabels(["Negative", "Positive"])
    st.pyplot(fig)

with col2:
    st.write("### WordClouds")
    col_pos, col_neg = st.columns(2)

    # Positive WordCloud
    with col_pos:
        st.write("✅ Positive Reviews")
        pos_text = " ".join(df[df['sentiment'] == 1]['review'].values)
        wordcloud_pos = WordCloud(width=400, height=300, background_color="white").generate(pos_text)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(wordcloud_pos, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

    # Negative WordCloud
    with col_neg:
        st.write("❌ Negative Reviews")
        neg_text = " ".join(df[df['sentiment'] == 0]['review'].values)
        wordcloud_neg = WordCloud(width=400, height=300, background_color="white").generate(neg_text)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(wordcloud_neg, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)