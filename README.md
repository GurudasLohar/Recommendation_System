# Content-Based Movie Recommendation System

## Live Demo
**Deployed URL:** https://recommendation-system-1q3h.onrender.com

## GitHub Repository
https://github.com/GurudasLohar/Recommendation_System

## Dataset
movies.csv — sample movie titles + overview text (30 movies).  
Educational sample similar to Kaggle/TMDB movie datasets  
(title + text description for content-based filtering).

## Project structure
```
app.py                 # Streamlit UI
preprocess_and_model.py
movies.csv
requirements.txt
README.md
```

## What it does
1. Load movie data
2. Clean overview text (lowercase, remove punctuation, stopwords)
3. TF-IDF vectorization (max_features=500, ngram_range=(1,2))
4. Cosine similarity between all movies
5. recommend(title, top_n) returns similar movies
6. Streamlit dropdown + button UI

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Git & GitHub
```
git init
git add .
git commit -m "content-based movie recommender"
git remote add origin https://github.com/GurudasLohar/Recommendation_System.git
git branch -M main
git push -u origin main
```

## Deploy on Render
1. Create account on render.com
2. New → Web Service → connect GitHub repo
3. Settings:
   - Build command: `pip install -r requirements.txt`
   - Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Deploy

**Live app:** https://recommendation-system-1q3h.onrender.com

