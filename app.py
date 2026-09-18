# Streamlit content-based movie recommender

import re
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Movie Recommender")
st.write("Content-based recommendations using TF-IDF + cosine similarity")

df = pd.read_csv("movies.csv")

STOP = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "must", "can", "this", "that", "these", "those", "i", "you", "he", "she",
    "it", "we", "they", "my", "your", "his", "her", "its", "our", "their",
    "with", "from", "by", "as", "not", "no", "so", "if", "then", "than",
    "too", "very", "just", "about", "into", "over", "after", "also", "only",
    "other", "new", "more", "most", "such", "own", "same"
}

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [w for w in text.split() if w not in STOP and len(w) > 1]
    return " ".join(tokens)

df["clean_text"] = df["overview"].apply(clean_text)

tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df["clean_text"])
sim_matrix = cosine_similarity(tfidf_matrix)
indices = pd.Series(df.index, index=df["title"])

def recommend(item_name, top_n=5):
    idx = indices[item_name]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top = scores[1: top_n + 1]
    return [(df["title"].iloc[i], round(float(score), 3)) for i, score in top]

movie = st.selectbox("Select a movie", df["title"].tolist())
top_n = st.slider("Number of recommendations", 3, 10, 5)

if st.button("Recommend"):
    results = recommend(movie, top_n)
    st.subheader("Recommended movies")
    for title, score in results:
        st.write(f"- **{title}** (similarity: {score})")
