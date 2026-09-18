# Content-Based Movie Recommendation System

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

## Git & GitHub (Task 7)
```
git init
git add .
git commit -m "content-based movie recommender"
# create repo on GitHub, then:
git remote add origin https://github.com/<your-username>/<repo>.git
git branch -M main
git push -u origin main
```

## Deploy on Render (Task 8)
1. Create account on render.com
2. New → Web Service → connect GitHub repo
3. Settings:
   - Build command: `pip install -r requirements.txt`
   - Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Deploy and test the live URL

## Notes
- Content-based only (no collaborative filtering)
- Cosine similarity measures closeness of TF-IDF vectors
- Keep UI simple as required
