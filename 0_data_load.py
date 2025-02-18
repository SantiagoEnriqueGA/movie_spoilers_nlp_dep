import pandas as pd
import os

# Load data
# -----------------------------------------------------------------------------
reviews_df = pd.read_json('data/raw/IMDB_reviews.json', lines=True)
movies_df = pd.read_json('data/raw/IMDB_movie_details.json', lines=True)

# Preprocess data
# -----------------------------------------------------------------------------
reviews_df = reviews_df.drop_duplicates("review_text").sample(frac=1)       # Drop duplicates and shuffle rows
reviews_df['review_date'] = pd.to_datetime(reviews_df['review_date'])       # Convert review_date to datetime
reviews_df['user_id'] = reviews_df['user_id'].astype('category').cat.codes  # Encode user_id as categories

movies_df['release_date'] = pd.to_datetime(movies_df['release_date'], format='mixed') # Convert release_date to datetime

# Save data
# -----------------------------------------------------------------------------
os.makedirs('data/processed', exist_ok=True)
reviews_df.to_csv('data/processed/reviews.csv', index=False)
movies_df.to_csv('data/processed/movie_details.csv', index=False)

# Print data shapes and heads for verification
# -----------------------------------------------------------------------------
print("Reviews DataFrame Shape:", reviews_df.shape)
print(reviews_df.head())
print("\nMovies DataFrame Shape:", movies_df.shape)
print(movies_df.head())