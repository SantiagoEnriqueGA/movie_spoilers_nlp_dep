import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast
import numpy as np
from collections import Counter
import re
from datetime import datetime

# Set some styling options
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

# Create output directory for plots
if not os.path.exists('eda/initial'):
    os.makedirs('eda/initial')

# Load data
# -----------------------------------------------------------------------------
reviews_df = pd.read_csv('data/processed/reviews.csv')
movies_df = pd.read_csv('data/processed/movie_details.csv')

# Convert dates and handle data types
reviews_df['review_date'] = pd.to_datetime(reviews_df['review_date'], errors='coerce')
movies_df['release_date'] = pd.to_datetime(movies_df['release_date'], errors='coerce')

# Convert genre strings to actual lists if they're stored as strings
if isinstance(movies_df['genre'].iloc[0], str):
    movies_df['genre'] = movies_df['genre'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Parse duration to minutes (assuming format like "2h 30m" or "1h" or "45m")
def parse_duration(duration_str):
    if pd.isna(duration_str):
        return np.nan
    
    hours = 0
    minutes = 0
    
    if 'h' in duration_str:
        hours = int(re.search(r'(\d+)h', duration_str).group(1))
    if 'm' in duration_str:
        minutes = int(re.search(r'(\d+)m', duration_str).group(1))
    
    return hours * 60 + minutes

movies_df['duration_minutes'] = movies_df['duration'].apply(parse_duration)

# Display basic information
# -----------------------------------------------------------------------------
print("Reviews DataFrame Shape:", reviews_df.shape)
print("Movies DataFrame Shape:", movies_df.shape)

print("\nReviews DataFrame Info:")
reviews_df.info()

print("\nMovies DataFrame Info:")
movies_df.info()

# Basic analysis
# -----------------------------------------------------------------------------
print("\nReviews Summary Statistics:")
reviews_summary = reviews_df.describe(include='all')
print(reviews_summary)

print("\nMovies Summary Statistics:")
movies_summary = movies_df.describe(include='all')
print(movies_summary)

# Missing values analysis
# -----------------------------------------------------------------------------
print("\nMissing Values in Reviews DataFrame:")
print(reviews_df.isnull().sum())

print("\nMissing Values in Movies DataFrame:")
print(movies_df.isnull().sum())

# Temporal analysis
# -----------------------------------------------------------------------------
# Reviews over time
plt.figure(figsize=(14, 8))
reviews_by_month = reviews_df.resample('ME', on='review_date').size()
ax = reviews_by_month.plot(kind='line')
plt.title('Number of Reviews Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Reviews')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('eda/initial/reviews_over_time.png')
plt.close()

# Reviews by day of week
plt.figure(figsize=(10, 6))
reviews_df['day_of_week'] = reviews_df['review_date'].dt.day_name()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
sns.countplot(x='day_of_week', data=reviews_df, order=day_order)
plt.title('Reviews by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('eda/initial/reviews_by_day.png')
plt.close()

# Movie releases by year
if not movies_df['release_date'].isnull().all():
    plt.figure(figsize=(14, 8))
    movies_df['release_year'] = movies_df['release_date'].dt.year
    releases_by_year = movies_df['release_year'].value_counts().sort_index()
    releases_by_year.plot(kind='bar')
    plt.title('Movie Releases by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Movies')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('eda/initial/releases_by_year.png')
    plt.close()

# Movie Rating Analysis
# -----------------------------------------------------------------------------
# Distribution of movie ratings
plt.figure(figsize=(10, 6))
sns.histplot(movies_df['rating'].dropna(), bins=20, kde=True)
plt.title('Distribution of Movie Ratings')
plt.xlabel('Movie Rating')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--')
plt.savefig('eda/initial/ratings_distribution_by_movie.png')
plt.close()

# Distribution of review ratings
plt.figure(figsize=(10, 6))
sns.countplot(x='rating', data=reviews_df)
plt.title('Distribution of Review Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--')
plt.savefig('eda/initial/ratings_distribution_by_review.png')
plt.close()

# Genre Analysis
# -----------------------------------------------------------------------------
# Extract all genres from the lists and count them
all_genres = []
for genres in movies_df['genre'].dropna():
    if isinstance(genres, list):
        all_genres.extend(genres)
    else:
        continue  # Skip if not a list

genre_counts = pd.Series(all_genres).value_counts()

# Plot top 15 genres
plt.figure(figsize=(14, 8))
genre_counts.head(15).plot(kind='bar')
plt.title('Top 15 Movie Genres')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('eda/initial/top_genres.png')
plt.close()

# Movie Duration Analysis
# -----------------------------------------------------------------------------
if not movies_df['duration_minutes'].isnull().all():
    plt.figure(figsize=(12, 6))
    sns.histplot(movies_df['duration_minutes'].dropna(), bins=30, kde=True)
    plt.title('Distribution of Movie Durations')
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--')
    plt.savefig('eda/initial/duration_distribution.png')
    plt.close()

# Correlation between movie duration and rating
if not movies_df['duration_minutes'].isnull().all():
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='duration_minutes', y='rating', data=movies_df.dropna(subset=['duration_minutes', 'rating']))
    plt.title('Movie Rating vs Duration')
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Rating')
    plt.grid(True)
    plt.savefig('eda/initial/duration_vs_rating.png')
    plt.close()

# Review Text Analysis
# -----------------------------------------------------------------------------
# Review length distribution
reviews_df['review_length'] = reviews_df['review_text'].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)

plt.figure(figsize=(12, 6))
sns.histplot(reviews_df['review_length'], bins=50, kde=True)
plt.title('Distribution of Review Lengths')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.xlim(0, reviews_df['review_length'].quantile(0.95))  # Limit to 95th percentile to avoid outliers
plt.grid(axis='y', linestyle='--')
plt.savefig('eda/initial/review_length_distribution.png')
plt.close()

# Review length vs rating
plt.figure(figsize=(10, 6))
sns.boxplot(x='rating', y='review_length', data=reviews_df)
plt.title('Review Length by Rating')
plt.xlabel('Rating')
plt.ylabel('Review Length (words)')
plt.ylim(0, reviews_df['review_length'].quantile(0.95))  # Limit to 95th percentile
plt.grid(axis='y', linestyle='--')
plt.savefig('eda/initial/review_length_by_rating.png')
plt.close()

# Spoiler Analysis
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
spoiler_counts = reviews_df['is_spoiler'].value_counts()
spoiler_counts.plot(kind='pie', autopct='%1.1f%%', explode=[0, 0.1], labels=['Non-Spoiler', 'Spoiler'])
plt.title('Proportion of Spoiler Reviews')
plt.ylabel('')  # Remove the y-label
plt.savefig('eda/initial/spoiler_proportion.png')
plt.close()

# Spoiler rate by rating
plt.figure(figsize=(12, 6))
spoiler_by_rating = reviews_df.groupby('rating')['is_spoiler'].mean() * 100
spoiler_by_rating.plot(kind='bar')
plt.title('Spoiler Rate by Rating')
plt.xlabel('Rating')
plt.ylabel('Spoiler Rate (%)')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig('eda/initial/spoiler_rate_by_rating.png')
plt.close()

# User Analysis
# -----------------------------------------------------------------------------
# User review count distribution
user_review_counts = reviews_df['user_id'].value_counts()

plt.figure(figsize=(12, 6))
sns.histplot(user_review_counts, bins=50, kde=True)
plt.title('Distribution of Reviews per User')
plt.xlabel('Number of Reviews')
plt.ylabel('Number of Users')
plt.xlim(0, user_review_counts.quantile(0.95))  # Limit to 95th percentile
plt.grid(axis='y', linestyle='--')
plt.savefig('eda/initial/user_review_count_distribution.png')
plt.close()

# Top 10 most active users
plt.figure(figsize=(14, 6))
top_users = user_review_counts.head(10)
top_users.plot(kind='bar')
plt.title('Top 10 Most Active Users')
plt.xlabel('User ID')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig('eda/initial/user_review_count_top10.png')
plt.close()

# Genre-Rating Analysis
# -----------------------------------------------------------------------------
# Create a new dataframe with exploded genres
if 'genre' in movies_df.columns:
    exploded_genres_df = movies_df.explode('genre')[['movie_id', 'genre', 'rating']]
    
    # Average rating by genre
    genre_ratings = exploded_genres_df.groupby('genre')['rating'].agg(['mean', 'count'])
    genre_ratings = genre_ratings.sort_values('mean', ascending=False)
    genre_ratings = genre_ratings[genre_ratings['count'] >= 5]  # Filter genres with at least 5 movies
    
    plt.figure(figsize=(14, 8))
    # Fixed barplot without the xerr argument that was causing the error
    genre_ratings_reset = genre_ratings.reset_index()
    ax = sns.barplot(x='mean', y='genre', hue='genre', data=genre_ratings_reset, legend=False)
    plt.title('Average Rating by Genre')
    plt.xlabel('Average Rating')
    plt.ylabel('Genre')
    plt.xlim(movies_df['rating'].min() - 0.5, movies_df['rating'].max() + 0.5)
    plt.tight_layout()
    plt.savefig('eda/initial/avg_rating_by_genre.png')
    plt.close()
    
    # Add a separate plot to visualize the counts alongside ratings
    plt.figure(figsize=(14, 8))
    genre_ratings = genre_ratings.sort_values('count', ascending=False)
    sns.barplot(x='count', y='genre', hue='genre', data=genre_ratings_reset, legend=False)
    plt.title('Number of Movies by Genre')
    plt.xlabel('Number of Movies')
    plt.ylabel('Genre')
    plt.tight_layout()
    plt.savefig('eda/initial/movie_count_by_genre.png')
    plt.close()

print("EDA completed and plots saved to the 'eda/initial' directory.")