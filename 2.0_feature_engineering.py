import os
import pandas as pd
import numpy as np
import re
import logging
import textstat
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize, ne_chunk, pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
import multiprocessing as mp
from datetime import datetime
import time
from joblib import Parallel, delayed
from multiprocessing import Pool, cpu_count

# Version Indicator
VERSION = 'v3'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=f'logs/feature_engineering_{VERSION}.log',
    filemode='w'
)
logger = logging.getLogger('feature_engineering')

# Console handler for logging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Custom transformers for pipeline
class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, use_keywords=True, n_jobs=-1):
        self.use_keywords = use_keywords
        self.n_jobs = n_jobs
        self.sid = SentimentIntensityAnalyzer()
        # Spoiler-related keywords
        self.keywords = [
            'spoiler', 'reveals', 'plot twist', 'ending', 'dies', 'death', 'killer', 
            'murderer', 'secret', 'betrayal', 'identity', 'truth', 'hidden', 'unveiled',
            'climax', 'finale', 'conclusion', 'resolution', 'twist', 'surprise', 
            'unexpected', 'shock', 'disclosure', 'unmask', 'revelation', 'expose', 
            'uncover', 'spoils', 'giveaway', 'leak', 'spoiling', 'foretell', 'foreshadow'
        ]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        logger.info("Extracting text features in parallel...")
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process_text)(text) for text in X
        )
        
        # Define column names
        columns = ['word_count', 'avg_word_length', 'num_sentences', 'compound_sentiment_score']
        if self.use_keywords:
            columns.extend([f'has_{keyword}' for keyword in self.keywords])
            
        return pd.DataFrame(results, columns=columns)
    
    def _process_text(self, text):
        try:
            word_count = len(word_tokenize(text))
            avg_word_len = self._avg_word_length(text)
            num_sentences = len(sent_tokenize(text))
            compound_sentiment = self.sid.polarity_scores(text)['compound']
            
            features = [word_count, avg_word_len, num_sentences, compound_sentiment]
            
            if self.use_keywords:
                keyword_flags = [int(keyword in text.lower()) for keyword in self.keywords]
                features.extend(keyword_flags)
            
            return features
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            default_features = [0, 0, 0, 0]
            if self.use_keywords:
                default_features.extend([0] * len(self.keywords))
            return default_features
    
    def _avg_word_length(self, text):
        try:
            words = word_tokenize(text)
            return sum(len(word) for word in words) / len(words)
        except Exception:
            return 0

class NamedEntityExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs
        self.entity_types = ['PERSON', 'ORGANIZATION', 'GPE', 'LOCATION', 'FACILITY', 'PRODUCT']
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        logger.info("Extracting named entities in parallel...")
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._extract_entities)(text) for text in X
        )
        columns = [f'entity_{entity_type}' for entity_type in self.entity_types]
        return pd.DataFrame(results, columns=columns)
    
    def _extract_entities(self, text):
        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            named_entities = ne_chunk(pos_tags)
            
            entity_counts = {entity_type: 0 for entity_type in self.entity_types}
            
            for entity in named_entities:
                if hasattr(entity, 'label') and entity.label() in self.entity_types:
                    entity_counts[entity.label()] += 1
            
            return [entity_counts[entity_type] for entity_type in self.entity_types]
        except Exception:
            return [0] * len(self.entity_types)

class ReadabilityMetricsExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        logger.info("Calculating readability metrics in parallel...")
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process_text)(text) for text in X
        )
        columns = ['flesch_reading_ease', 'flesch_kincaid_grade', 'smog_index']
        return pd.DataFrame(results, columns=columns)
    
    def _process_text(self, text):
        try:
            flesch_score = textstat.flesch_reading_ease(text)
            flesch_kincaid_score = textstat.flesch_kincaid_grade(text)
            smog_score = textstat.smog_index(text)
            return [flesch_score, flesch_kincaid_score, smog_score]
        except Exception as e:
            logger.error(f"Error calculating readability: {e}")
            return [0, 0, 0]

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        logger.info("Extracting date features...")
        dates = pd.to_datetime(X, errors='coerce')
        
        result = pd.DataFrame({
            'review_year': dates.dt.year,
            'review_month': dates.dt.month,
            'review_day': dates.dt.day
        })
        
        return result

class TfidfFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=5000, n_components=100, use_ngrams=True, use_hashing=False):
        self.max_features = max_features
        self.n_components = n_components
        self.use_ngrams = use_ngrams
        self.use_hashing = use_hashing

        if self.use_hashing:
            # HashingVectorizer is very fast and memory efficient, but it doesn't build a vocabulary.
            # Using TfidfTransformer afterwards recovers tf-idf weighting.
            self.vectorizer = HashingVectorizer(
                n_features=self.max_features, 
                ngram_range=(1, 3) if use_ngrams else (1, 1), 
                alternate_sign=False,
                norm=None
            )
            self.transformer = TfidfTransformer()
        else:
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features, 
                ngram_range=(1, 3) if use_ngrams else (1, 1)
            )
            self.transformer = None

        # Reduce n_iter for faster SVD computation (default is 5; set lower if acceptable)
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42, n_iter=5)

    def fit(self, X, y=None):
        logger.info(f"Fitting vectorizer with {self.max_features} features...")
        if self.use_hashing:
            # HashingVectorizer does not need to be fit, so we directly transform
            X_transformed = self.vectorizer.transform(X)
            X_transformed = self.transformer.fit_transform(X_transformed)
        else:
            X_transformed = self.vectorizer.fit_transform(X)
        
        logger.info(f"Fitting SVD with {self.n_components} components...")
        self.svd.fit(X_transformed)
        return self

    def transform(self, X):
        logger.info("Transforming text to vectorized features...")
        if self.use_hashing:
            X_transformed = self.vectorizer.transform(X)
            X_transformed = self.transformer.transform(X_transformed)
        else:
            X_transformed = self.vectorizer.transform(X)
        
        logger.info("Applying dimensionality reduction with SVD...")
        svd_matrix = self.svd.transform(X_transformed)
        
        result = pd.DataFrame(
            svd_matrix, 
            columns=[f'svd_{i}' for i in range(self.n_components)]
        )
        
        return result

class TopicModelingExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, n_topics=10, max_features=5000):
        self.n_topics = n_topics
        self.max_features = max_features
        self.tfidf = TfidfVectorizer(max_features=max_features)
        self.lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=10,
            learning_method='online',
            learning_offset=50.,
            random_state=42
        )
        
    def fit(self, X, y=None):
        logger.info(f"Fitting TF-IDF vectorizer for topic modeling with {self.max_features} features...")
        tfidf_matrix = self.tfidf.fit_transform(X)
        
        logger.info(f"Fitting LDA with {self.n_topics} topics...")
        self.lda.fit(tfidf_matrix)
        return self
    
    def transform(self, X):
        logger.info("Transforming text to topic features...")
        tfidf_matrix = self.tfidf.transform(X)
        
        logger.info("Extracting topic distributions...")
        topic_distributions = self.lda.transform(tfidf_matrix)
        
        # Create DataFrame with topic column names
        result = pd.DataFrame(
            topic_distributions,
            columns=[f'topic_{i}' for i in range(self.n_topics)]
        )
        
        return result

class CosineSimilarityExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, text_col1, text_col2, output_col_name, batch_size=100):
        self.text_col1 = text_col1
        self.text_col2 = text_col2
        self.output_col_name = output_col_name
        self.batch_size = batch_size
        self.tfidf_vectorizer = TfidfVectorizer()
        
    def fit(self, X, y=None):
        logger.info(f"Fitting vectorizer for {self.text_col1} and {self.text_col2}...")
        combined_text = X[self.text_col1].fillna('').tolist() + X[self.text_col2].fillna('').tolist()
        self.tfidf_vectorizer.fit(combined_text)
        return self
    
    def transform(self, X):
        logger.info(f"Computing cosine similarity between {self.text_col1} and {self.text_col2}...")
        tfidf_col1 = self.tfidf_vectorizer.transform(X[self.text_col1].fillna(''))
        tfidf_col2 = self.tfidf_vectorizer.transform(X[self.text_col2].fillna(''))
        
        cosine_similarities = self._compute_cosine_similarity_batched(tfidf_col1, tfidf_col2)
        
        result = pd.DataFrame({
            self.output_col_name: cosine_similarities
        })
        
        return result
    
    def _compute_cosine_similarity_batched(self, tfidf_matrix1, tfidf_matrix2):
        num_samples = tfidf_matrix1.shape[0]
        cosine_similarities = []

        for start_idx in range(0, num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_samples)
            # Compute cosine similarity for the current batch
            cosine_sim = cosine_similarity(tfidf_matrix1[start_idx:end_idx], tfidf_matrix2[start_idx:end_idx]).diagonal()
            cosine_similarities.append(cosine_sim)

        # Flatten the list of results into a single array
        return np.hstack(cosine_similarities)

def convert_duration_to_minutes(duration):
    """
    Converts a duration string in the format 'Xh Ymin' to minutes.
    Args:
        duration (str): The duration string to be converted.
    Returns:
        int: The duration in minutes.
    Examples:
        convert_duration_to_minutes('2h 30min')  # Returns: 150
    """
    match = re.match(r'(\d+)h (\d+)min', duration)  # Match the duration string
    if match:                                       # If the match is found
        hours = int(match.group(1))                 # Extract the hours
        minutes = int(match.group(2))               # Extract the minutes
        return hours * 60 + minutes                 # Convert to minutes
    else:
        return 0                                    # Return 0 if no match is found

def calculate_review_length_percentile(df):
    """
    Calculate the percentile rank of review length within each movie group.
    Args:
        df (pandas.DataFrame): DataFrame containing 'movie_id' and 'word_count' columns
    Returns:
        pandas.Series: Series containing the percentile ranks
    """
    return df.groupby('movie_id')['word_count'].apply(lambda x: x.rank(pct=True))


if __name__ == "__main__":
    # Set start time for overall execution
    total_start_time = time.time()
    logger.info("Starting feature engineering process...")

    # Load data
    logger.info("Loading data...")
    reviews_df = pd.read_csv('data/processed/reviews.csv')
    movies_df = pd.read_csv('data/processed/movie_details.csv')

    # # TESTING: use only first 1000 rows
    # reviews_df = reviews_df.head(1000)
    # movies_df = movies_df.head(1000)
    
    # 1. Process Text Features using pipeline
    # ---------------------------------------------------------------------------------------------
    logger.info("Processing text features with pipeline...")
    text_feature_start = time.time()
    
    text_pipeline = Pipeline([
        ('text_features', TextFeatureExtractor(use_keywords=True))
    ])
    
    text_features_df = text_pipeline.fit_transform(reviews_df['review_text'])
    reviews_df = pd.concat([reviews_df, text_features_df], axis=1)
    
    text_feature_end = time.time()
    logger.info(f"Text features processing completed in {text_feature_end - text_feature_start:.2f} seconds")
    
    # 2. Process Date Features
    # ---------------------------------------------------------------------------------------------
    logger.info("Processing date features...")
    date_feature_start = time.time()
    
    date_pipeline = Pipeline([
        ('date_features', DateFeatureExtractor())
    ])
    
    date_features_df = date_pipeline.fit_transform(reviews_df['review_date'])
    reviews_df = pd.concat([reviews_df, date_features_df], axis=1)
    
    date_feature_end = time.time()
    logger.info(f"Date features processing completed in {date_feature_end - date_feature_start:.2f} seconds")
    
    # 3. Process Movie Details
    # ---------------------------------------------------------------------------------------------
    logger.info("Processing movie details features...")
    movie_details_start = time.time()
    
    movies_df['duration_minutes'] = movies_df['duration'].apply(convert_duration_to_minutes)
    merged_df = pd.merge(reviews_df, movies_df, on='movie_id', how='inner')
    
    movie_details_end = time.time()
    logger.info(f"Movie details processing completed in {movie_details_end - movie_details_start:.2f} seconds")
    
    # 4. Statistical Features
    # ---------------------------------------------------------------------------------------------
    logger.info("Calculating statistical features...")
    stats_feature_start = time.time()
    
    rating_stats = reviews_df.groupby('movie_id')['rating'].agg(['mean', 'median', 'std']).reset_index()
    rating_stats.columns = ['movie_id', 'rating_mean', 'rating_median', 'rating_std']
    final_df = pd.merge(merged_df, rating_stats, on='movie_id', how='left')
    
    # Calculate review length percentile within the same movie
    logger.info("Calculating review length percentile...")
    percentiles = calculate_review_length_percentile(final_df)
    percentiles = percentiles.reset_index(level=0, drop=True)
    final_df['review_length_percentile'] = percentiles

    stats_feature_end = time.time()
    logger.info(f"Statistical features calculation completed in {stats_feature_end - stats_feature_start:.2f} seconds")
    
    # 5. Temporal Features: Calculate days between movie release and review
    # ---------------------------------------------------------------------------------------------
    logger.info("Calculating temporal features...")
    temporal_feature_start = time.time()
    
    try:
        final_df['release_date'] = pd.to_datetime(final_df['release_date'], errors='coerce')
        final_df['review_date'] = pd.to_datetime(final_df['review_date'], errors='coerce')
        final_df['days_since_release'] = (final_df['review_date'] - final_df['release_date']).dt.days
    except Exception as e:
        logger.error(f"Error calculating days since release: {e}")
        final_df['days_since_release'] = 0
    
    temporal_feature_end = time.time()
    logger.info(f"Temporal features calculation completed in {temporal_feature_end - temporal_feature_start:.2f} seconds")
    
    # 6. Named Entity Recognition
    # ---------------------------------------------------------------------------------------------
    logger.info("Extracting named entities...")
    ner_start = time.time()
    
    ner_pipeline = Pipeline([
        ('named_entities', NamedEntityExtractor())
    ])
    
    ner_features_df = ner_pipeline.fit_transform(final_df['review_text'])
    final_df = pd.concat([final_df, ner_features_df], axis=1)
    
    ner_end = time.time()
    logger.info(f"Named entity extraction completed in {ner_end - ner_start:.2f} seconds")
    
    # 7. Readability Metrics
    # ---------------------------------------------------------------------------------------------
    logger.info("Calculating readability metrics...")
    readability_start = time.time()
    
    readability_pipeline = Pipeline([
        ('readability', ReadabilityMetricsExtractor())
    ])
    
    readability_features_df = readability_pipeline.fit_transform(final_df['review_text'])
    final_df = pd.concat([final_df, readability_features_df], axis=1)
    
    readability_end = time.time()
    logger.info(f"Readability metrics calculation completed in {readability_end - readability_start:.2f} seconds")
    
    # 8. Cosine Similarity Features (including Review-Movie Title Similarity)
    # ---------------------------------------------------------------------------------------------
    logger.info("Computing cosine similarity features...")
    cosine_feature_start = time.time()
    
    # Define cosine similarity pipelines for different text pairs
    review_plot_similarity = CosineSimilarityExtractor(
        'review_text', 'plot_summary', 'cosine_sim_review_plot'
    )
    summary_synopsis_similarity = CosineSimilarityExtractor(
        'review_summary', 'plot_synopsis', 'cosine_sim_summary_synopsis'
    )
    
    # Apply transformations
    review_plot_sim_df = review_plot_similarity.fit_transform(final_df)
    summary_synopsis_sim_df = summary_synopsis_similarity.fit_transform(final_df)
    
    # Combine results
    final_df = pd.concat([
        final_df, 
        review_plot_sim_df, 
        summary_synopsis_sim_df,
    ], axis=1)
    
    cosine_feature_end = time.time()
    logger.info(f"Cosine similarity computation completed in {cosine_feature_end - cosine_feature_start:.2f} seconds")
    
    # 9. TF-IDF Features with Bigrams and Trigrams
    # ---------------------------------------------------------------------------------------------
    logger.info("Generating TF-IDF features with n-grams...")
    tfidf_feature_start = time.time()
    
    tfidf_pipeline = Pipeline([
        ('tfidf_features', TfidfFeatureExtractor(max_features=5000, n_components=100, use_ngrams=True, use_hashing=True))
    ])
    
    tfidf_features_df = tfidf_pipeline.fit_transform(final_df['review_text'])
    final_df = pd.concat([final_df, tfidf_features_df], axis=1)
    
    tfidf_feature_end = time.time()
    logger.info(f"TF-IDF feature generation completed in {tfidf_feature_end - tfidf_feature_start:.2f} seconds")
    
    # 10. Topic Modeling using LDA
    # ---------------------------------------------------------------------------------------------
    logger.info("Generating topic modeling features...")
    topic_modeling_start = time.time()
    
    topic_pipeline = Pipeline([
        ('topic_modeling', TopicModelingExtractor(n_topics=10, max_features=5000))
    ])
    
    topic_features_df = topic_pipeline.fit_transform(final_df['review_text'])
    final_df = pd.concat([final_df, topic_features_df], axis=1)
    
    topic_modeling_end = time.time()
    logger.info(f"Topic modeling feature generation completed in {topic_modeling_end - topic_modeling_start:.2f} seconds")
    
    # Save Results
    # ---------------------------------------------------------------------------------------------
    logger.info("Saving processed datasets...")
    save_start = time.time()
    
    os.makedirs(f'data/processed/{VERSION}', exist_ok=True)
    
    reviews_df.to_parquet(f'data/processed/{VERSION}/reviews_engineered.parquet', index=False)
    movies_df.to_parquet(f'data/processed/{VERSION}/movies_engineered.parquet', index=False)
    merged_df.to_parquet(f'data/processed/{VERSION}/merged.parquet', index=False)
    final_df.to_parquet(f'data/processed/{VERSION}/final_engineered.parquet', index=False)
    
    save_end = time.time()
    logger.info(f"Dataset saving completed in {save_end - save_start:.2f} seconds")
    
    # Report total execution time
    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time
    logger.info(f"Total feature engineering process completed in {total_execution_time:.2f} seconds")