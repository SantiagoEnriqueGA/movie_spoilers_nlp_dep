import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast
import numpy as np
from collections import Counter
import re
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# Set some styling options
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

# Create output directory for plots
if not os.path.exists('eda/engineered'):
    os.makedirs('eda/engineered')

# Load data
# -----------------------------------------------------------------------------
# Version Indicator
VERSION = 'v3'

# Load the data
try:
    df = pd.read_parquet(f'data/processed/v3/final_engineered.parquet')
    print(f"Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
except FileNotFoundError:
    print("Dataset file not found. Please check the file path.")
    raise

# Basic data information
# -----------------------------------------------------------------------------
print("\nBasic Information:")
print(f"Total samples: {df.shape[0]}")
print(f"Total features: {df.shape[1]}")
print(f"Target variable distribution: {df['is_spoiler'].value_counts(normalize=True)}")

# Missing values analysis
# -----------------------------------------------------------------------------
missing_values = df.isnull().sum().sort_values(ascending=False)
missing_values = missing_values[missing_values > 0]
missing_percent = (missing_values / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing Values': missing_values,
    'Missing Percent': missing_percent
})

print("\nMissing Values Analysis:")
print(missing_df)

if len(missing_df) > 0:
    plt.figure(figsize=(14, 8))
    sns.barplot(x=missing_df.index, y='Missing Percent', data=missing_df)
    plt.title('Percentage of Missing Values by Feature')
    plt.xlabel('Features')
    plt.ylabel('Missing Percentage')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('eda/engineered/missing_values.png')
    plt.close()

# Numerical feature distribution
# -----------------------------------------------------------------------------
# Get numerical features
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
# Exclude SVD features for basic analysis (too many)
numerical_features = [f for f in numerical_features if not f.startswith('svd_')]
numerical_features = [f for f in numerical_features if not f.startswith('topic_')]

# Histograms for numerical features
for i in range(0, len(numerical_features), 9):
    features_chunk = numerical_features[i:i+9]
    num_features = len(features_chunk)
    num_rows = (num_features + 2) // 3  # Ceiling division to get number of rows
    
    fig, axes = plt.subplots(num_rows, 3, figsize=(18, num_rows * 4))
    axes = axes.flatten()
    
    for j, feature in enumerate(features_chunk):
        sns.histplot(df[feature].dropna(), kde=True, ax=axes[j])
        axes[j].set_title(f'Distribution of {feature}')
        axes[j].grid(True)
    
    # Turn off unused subplots
    for j in range(num_features, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'eda/engineered/numerical_dist_{i//9+1}.png')
    plt.close()

# Target correlation analysis
# -----------------------------------------------------------------------------
# Convert boolean target to integer for correlation
if df['is_spoiler'].dtype == 'bool':
    df['is_spoiler'] = df['is_spoiler'].astype(int)

# Calculate correlation with target
target_corr = df[numerical_features].corrwith(df['is_spoiler']).sort_values(ascending=False)

plt.figure(figsize=(16, 10))
sns.barplot(x=target_corr.values, y=target_corr.index)
plt.title('Correlation with Target (is_spoiler)')
plt.xlabel('Correlation Coefficient')
plt.tight_layout()
plt.savefig('eda/engineered/target_correlation.png')
plt.close()

# Top 10 positively correlated features
plt.figure(figsize=(14, 8))
top_pos_corr = target_corr.head(10)
sns.barplot(x=top_pos_corr.values, y=top_pos_corr.index, palette='Reds_r')
plt.title('Top 10 Positively Correlated Features with is_spoiler')
plt.xlabel('Correlation Coefficient')
plt.tight_layout()
plt.savefig('eda/engineered/top_pos_correlation.png')
plt.close()

# Top 10 negatively correlated features
plt.figure(figsize=(14, 8))
top_neg_corr = target_corr.tail(10)
sns.barplot(x=top_neg_corr.values, y=top_neg_corr.index, palette='Blues')
plt.title('Top 10 Negatively Correlated Features with is_spoiler')
plt.xlabel('Correlation Coefficient')
plt.tight_layout()
plt.savefig('eda/engineered/top_neg_correlation.png')
plt.close()

# Feature relationships
# -----------------------------------------------------------------------------
# Select top correlated features for pair plots
top_corr_features = list(target_corr.head(5).index) + list(target_corr.tail(5).index)
top_corr_features = [f for f in top_corr_features if f != 'is_spoiler']

# Analyze relationships between top features and target
df_sample = df.sample(min(10000, len(df)))
pair_df = df_sample[top_corr_features + ['is_spoiler']]

plt.figure(figsize=(20, 15))
sns.pairplot(pair_df, hue='is_spoiler', palette='Set1')
plt.tight_layout()
plt.savefig('eda/engineered/feature_pairplot.png')
plt.close()

# Spoiler keywords analysis
# -----------------------------------------------------------------------------
# Gather all the spoiler-related keyword features
spoiler_features = [col for col in df.columns if col.startswith('has_')]
spoiler_counts = df[spoiler_features].sum().sort_values(ascending=False)

plt.figure(figsize=(16, 10))
sns.barplot(x=spoiler_counts.values, y=spoiler_counts.index)
plt.title('Frequency of Spoiler Keywords')
plt.xlabel('Count')
plt.ylabel('Keyword')
plt.tight_layout()
plt.savefig('eda/engineered/spoiler_keyword_freq.png')
plt.close()

# Analyze spoiler keywords relationship with target
spoiler_keyword_corr = df[spoiler_features].corrwith(df['is_spoiler']).sort_values(ascending=False)

plt.figure(figsize=(16, 10))
sns.barplot(x=spoiler_keyword_corr.values, y=spoiler_keyword_corr.index, palette='RdBu_r')
plt.title('Correlation of Spoiler Keywords with Target')
plt.xlabel('Correlation Coefficient')
plt.axvline(x=0, color='black', linestyle='--')
plt.tight_layout()
plt.savefig('eda/engineered/spoiler_keyword_corr.png')
plt.close()

# Sentiment analysis
# -----------------------------------------------------------------------------
if 'compound_sentiment_score' in df.columns:
    plt.figure(figsize=(12, 8))
    sns.histplot(data=df, x='compound_sentiment_score', hue='is_spoiler', kde=True, palette='Set1')
    plt.title('Sentiment Score Distribution by Spoiler Status')
    plt.xlabel('Compound Sentiment Score')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('eda/engineered/sentiment_by_spoiler.png')
    plt.close()
    
    # Sentiment by rating
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='rating_x', y='compound_sentiment_score')
    plt.title('Sentiment Score by Rating')
    plt.xlabel('Rating')
    plt.ylabel('Compound Sentiment Score')
    plt.tight_layout()
    plt.savefig('eda/engineered/sentiment_by_rating.png')
    plt.close()

# Text complexity features
# -----------------------------------------------------------------------------
complexity_features = [
    'word_count', 'avg_word_length', 'num_sentences', 
    'flesch_reading_ease', 'flesch_kincaid_grade', 'smog_index'
]

if all(feature in df.columns for feature in complexity_features):
    plt.figure(figsize=(18, 12))
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(complexity_features):
        sns.boxplot(data=df, x='is_spoiler', y=feature, ax=axes[i])
        axes[i].set_title(f'{feature} by Spoiler Status')
        axes[i].set_xlabel('Is Spoiler')
        
    plt.tight_layout()
    plt.savefig('eda/engineered/text_complexity_by_spoiler.png')
    plt.close()
    
    # Correlation heatmap for text complexity features
    plt.figure(figsize=(12, 10))
    complexity_corr = df[complexity_features + ['is_spoiler']].corr()
    sns.heatmap(complexity_corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation of Text Complexity Features')
    plt.tight_layout()
    plt.savefig('eda/engineered/text_complexity_corr.png')
    plt.close()

# SVD and Topic analysis
# -----------------------------------------------------------------------------
# SVD features
svd_features = [col for col in df.columns if col.startswith('svd_')]
topic_features = [col for col in df.columns if col.startswith('topic_')]

# PCA on SVD features (for visualization)
if len(svd_features) > 0:
    # Sample for performance if dataset is large
    sample_size = min(20000, df.shape[0])
    df_sample = df.sample(sample_size, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    svd_scaled = scaler.fit_transform(df_sample[svd_features])
    
    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    svd_2d = pca.fit_transform(svd_scaled)
    
    # Create a dataframe for plotting
    svd_df = pd.DataFrame({
        'SVD_PCA1': svd_2d[:, 0],
        'SVD_PCA2': svd_2d[:, 1],
        'is_spoiler': df_sample['is_spoiler']
    })
    
    # Plot PCA results
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=svd_df, x='SVD_PCA1', y='SVD_PCA2', hue='is_spoiler', alpha=0.5, palette='Set1')
    plt.title('PCA of SVD Features Colored by Spoiler Status')
    plt.xlabel(f'PCA1 (Variance Explained: {pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PCA2 (Variance Explained: {pca.explained_variance_ratio_[1]:.2%})')
    plt.tight_layout()
    plt.savefig('eda/engineered/svd_pca_by_spoiler.png')
    plt.close()

# Topic distribution
if len(topic_features) > 0:
    # Average topic weights by spoiler and non-spoiler
    topic_avg_by_spoiler = df.groupby('is_spoiler')[topic_features].mean()
    
    plt.figure(figsize=(14, 8))
    topic_avg_by_spoiler.T.plot(kind='bar')
    plt.title('Average Topic Weights by Spoiler Status')
    plt.xlabel('Topic')
    plt.ylabel('Average Weight')
    plt.legend(title='Is Spoiler')
    plt.tight_layout()
    plt.savefig('eda/engineered/topic_avg_by_spoiler.png')
    plt.close()
    
    # Topic correlation with target
    topic_corr = df[topic_features].corrwith(df['is_spoiler']).sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=topic_corr.values, y=topic_corr.index, palette='RdBu_r')
    plt.title('Correlation of Topics with Spoiler Status')
    plt.xlabel('Correlation Coefficient')
    plt.axvline(x=0, color='black', linestyle='--')
    plt.tight_layout()
    plt.savefig('eda/engineered/topic_correlation.png')
    plt.close()

# Entity analysis
# -----------------------------------------------------------------------------
entity_features = [col for col in df.columns if col.startswith('entity_')]

if len(entity_features) > 0:
    entity_counts = df[entity_features].sum().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=entity_counts.values, y=entity_counts.index)
    plt.title('Frequency of Named Entities')
    plt.xlabel('Count')
    plt.ylabel('Entity Type')
    plt.tight_layout()
    plt.savefig('eda/engineered/entity_counts.png')
    plt.close()
    
    # Entity correlation with target
    entity_corr = df[entity_features].corrwith(df['is_spoiler']).sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=entity_corr.values, y=entity_corr.index, palette='RdBu_r')
    plt.title('Correlation of Entity Types with Spoiler Status')
    plt.xlabel('Correlation Coefficient')
    plt.axvline(x=0, color='black', linestyle='--')
    plt.tight_layout()
    plt.savefig('eda/engineered/entity_correlation.png')
    plt.close()
    
    # Entity presence by spoiler
    plt.figure(figsize=(14, 8))
    entity_by_spoiler = df.groupby('is_spoiler')[entity_features].mean()
    entity_by_spoiler.T.plot(kind='bar')
    plt.title('Entity Presence by Spoiler Status')
    plt.xlabel('Entity Type')
    plt.ylabel('Average Presence')
    plt.legend(title='Is Spoiler')
    plt.tight_layout()
    plt.savefig('eda/engineered/entity_by_spoiler.png')
    plt.close()

# Similarity score analysis
# -----------------------------------------------------------------------------
sim_features = ['cosine_sim_review_plot', 'cosine_sim_summary_synopsis']

if all(feature in df.columns for feature in sim_features):
    plt.figure(figsize=(14, 6))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for i, feature in enumerate(sim_features):
        sns.histplot(data=df, x=feature, hue='is_spoiler', kde=True, ax=axes[i])
        axes[i].set_title(f'{feature} by Spoiler Status')
        axes[i].set_xlabel('Similarity Score')
        axes[i].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('eda/engineered/similarity_scores_by_spoiler.png')
    plt.close()
    
    # Correlation with other features
    sim_corr = df[complexity_features + sim_features].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(sim_corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation of Similarity Scores with Text Complexity')
    plt.tight_layout()
    plt.savefig('eda/engineered/similarity_complexity_corr.png')
    plt.close()

# Temporal features analysis
# -----------------------------------------------------------------------------
temporal_features = ['review_year', 'review_month', 'review_day', 'days_since_release']

if all(feature in df.columns for feature in temporal_features):
    # Spoiler distribution by year
    plt.figure(figsize=(14, 8))
    year_spoiler = df.groupby('review_year')['is_spoiler'].mean() * 100
    year_counts = df.groupby('review_year').size()
    
    # Create a figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax2 = ax1.twinx()
    
    # Plot spoiler percentage on first axis
    ax1.plot(year_spoiler.index, year_spoiler.values, 'b-', linewidth=2)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Spoiler Percentage (%)', color='b')
    ax1.tick_params('y', colors='b')
    
    # Plot review count on second axis
    ax2.bar(year_counts.index, year_counts.values, alpha=0.3, color='gray')
    ax2.set_ylabel('Number of Reviews', color='gray')
    ax2.tick_params('y', colors='gray')
    
    plt.title('Spoiler Percentage and Review Count by Year')
    plt.tight_layout()
    plt.savefig('eda/engineered/spoiler_by_year.png')
    plt.close()
    
    # Spoiler distribution by month
    plt.figure(figsize=(12, 6))
    month_spoiler = df.groupby('review_month')['is_spoiler'].mean() * 100
    month_counts = df.groupby('review_month').size()
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    ax1.plot(month_spoiler.index, month_spoiler.values, 'g-', linewidth=2)
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Spoiler Percentage (%)', color='g')
    ax1.tick_params('y', colors='g')
    
    ax2.bar(month_counts.index, month_counts.values, alpha=0.3, color='gray')
    ax2.set_ylabel('Number of Reviews', color='gray')
    ax2.tick_params('y', colors='gray')
    
    plt.title('Spoiler Percentage and Review Count by Month')
    plt.xticks(range(1, 13))
    plt.tight_layout()
    plt.savefig('eda/engineered/spoiler_by_month.png')
    plt.close()
    
    # Days since release analysis
    if 'days_since_release' in df.columns:
        # Group by bins of days since release
        df['days_since_release_bin'] = pd.cut(df['days_since_release'], 
                                            bins=[-1, 7, 30, 90, 180, 365, float('inf')],
                                            labels=['First Week', 'First Month', '1-3 Months', 
                                                    '3-6 Months', '6-12 Months', 'Over a Year'])
        
        plt.figure(figsize=(14, 8))
        release_spoiler = df.groupby('days_since_release_bin')['is_spoiler'].mean() * 100
        release_counts = df.groupby('days_since_release_bin').size()
        
        fig, ax1 = plt.subplots(figsize=(14, 8))
        ax2 = ax1.twinx()
        
        release_spoiler.plot(kind='line', marker='o', color='purple', ax=ax1)
        ax1.set_xlabel('Time Since Release')
        ax1.set_ylabel('Spoiler Percentage (%)', color='purple')
        ax1.tick_params('y', colors='purple')
        
        release_counts.plot(kind='bar', alpha=0.3, color='gray', ax=ax2)
        ax2.set_ylabel('Number of Reviews', color='gray')
        ax2.tick_params('y', colors='gray')
        
        plt.title('Spoiler Percentage and Review Count by Time Since Release')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('eda/engineered/spoiler_by_release_time.png')
        plt.close()

# Movie metadata analysis
# -----------------------------------------------------------------------------
if 'genre' in df.columns:
    # Convert string representation of list to actual list if necessary
    if isinstance(df['genre'].iloc[0], str):
        df['genre'] = df['genre'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Explode genre list to get one row per genre
    genre_df = df.explode('genre')[['genre', 'is_spoiler']]
    
    # Calculate spoiler percentage by genre
    genre_spoiler = genre_df.groupby('genre')['is_spoiler'].agg(['mean', 'count'])
    genre_spoiler['mean'] = genre_spoiler['mean'] * 100  # Convert to percentage
    genre_spoiler = genre_spoiler.sort_values('mean', ascending=False)
    
    # Filter to genres with sufficient samples
    genre_spoiler = genre_spoiler[genre_spoiler['count'] >= 100]
    
    plt.figure(figsize=(16, 10))
    fig, ax1 = plt.subplots(figsize=(16, 10))
    ax2 = ax1.twinx()
    
    genre_spoiler['mean'].plot(kind='bar', position=0, width=0.4, color='purple', ax=ax1)
    ax1.set_xlabel('Genre')
    ax1.set_ylabel('Spoiler Percentage (%)', color='purple')
    ax1.tick_params('y', colors='purple')
    
    genre_spoiler['count'].plot(kind='bar', position=1, width=0.4, color='gray', alpha=0.6, ax=ax2)
    ax2.set_ylabel('Number of Reviews', color='gray')
    ax2.tick_params('y', colors='gray')
    
    plt.title('Spoiler Percentage and Review Count by Genre')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('eda/engineered/spoiler_by_genre.png')
    plt.close()

# Review length and user behavior analysis
# -----------------------------------------------------------------------------
if 'review_length_percentile' in df.columns:
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='is_spoiler', y='review_length_percentile')
    plt.title('Review Length Percentile by Spoiler Status')
    plt.xlabel('Is Spoiler')
    plt.ylabel('Review Length Percentile')
    plt.tight_layout()
    plt.savefig('eda/engineered/review_length_by_spoiler.png')
    plt.close()

# User rating analysis
# -----------------------------------------------------------------------------
if 'rating_x' in df.columns:
    plt.figure(figsize=(12, 8))
    rating_spoiler = df.groupby('rating_x')['is_spoiler'].mean() * 100
    rating_counts = df.groupby('rating_x').size()
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()
    
    rating_spoiler.plot(kind='line', marker='o', color='red', ax=ax1)
    ax1.set_xlabel('Rating')
    ax1.set_ylabel('Spoiler Percentage (%)', color='red')
    ax1.tick_params('y', colors='red')
    
    rating_counts.plot(kind='bar', alpha=0.3, color='gray', ax=ax2)
    ax2.set_ylabel('Number of Reviews', color='gray')
    ax2.tick_params('y', colors='gray')
    
    plt.title('Spoiler Percentage and Review Count by Rating')
    plt.tight_layout()
    plt.savefig('eda/engineered/spoiler_by_rating.png')
    plt.close()

# Generate feature importance summary
# -----------------------------------------------------------------------------
# Calculate feature importance based on correlation with target
feature_importance = abs(df.select_dtypes(include=['float64', 'int64']).corrwith(df['is_spoiler'])).sort_values(ascending=False)
feature_importance = feature_importance.head(30)  # Top 30 features

plt.figure(figsize=(16, 12))
sns.barplot(x=feature_importance.values, y=feature_importance.index)
plt.title('Feature Importance (Based on Correlation Magnitude)')
plt.xlabel('Absolute Correlation with Target')
plt.tight_layout()
plt.savefig('eda/engineered/feature_importance.png')
plt.close()

print("Post-Feature Engineering EDA completed. All visualizations saved to 'eda/engineered' directory.")