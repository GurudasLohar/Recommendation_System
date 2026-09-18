# Content-based recommendation - preprocessing & similarity
# Dataset: movies.csv (sample movie title + overview)
# Similar style to Kaggle/TMDB movie datasets

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

# Task 1: Load & understand
df = pd.read_csv("movies.csv")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())
print("Text column used: overview")

# Task 2: Text preprocessing
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [w for w in text.split() if w not in STOP and len(w) > 1]
    return " ".join(tokens)

df["clean_text"] = df["overview"].apply(clean_text)
print("\nClean text sample:")
print(df[["title", "clean_text"]].head(3))

# Task 3: TF-IDF
tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df["clean_text"])
print("\nTF-IDF shape:", tfidf_matrix.shape)

# Task 4: Cosine similarity
# cosine similarity = how close two vectors are (good for text)
sim_matrix = cosine_similarity(tfidf_matrix)
print("Similarity matrix shape:", sim_matrix.shape)

# Task 5: Recommend function
indices = pd.Series(df.index, index=df["title"]).drop_duplicates()

def recommend(item_name, top_n=5):
    if item_name not in indices:
        return ["Item not found"]
    idx = indices[item_name]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top = scores[1: top_n + 1]
    return [(df["title"].iloc[i], round(score, 3)) for i, score in top]

print("\nRecommendations for 'Inception':")
print(recommend("Inception"))
print("\nRecommendations for 'The Godfather':")
print(recommend("The Godfather"))
print("\nRecommendations for 'Toy Story':")
print(recommend("Toy Story"))
