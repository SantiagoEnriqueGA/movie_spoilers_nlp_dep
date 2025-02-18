# Predicting Movie Spoilers from Natural Language  

This project explores the prediction of movie spoilers using natural language processing (NLP) techniques and the IMDB Spoiler Dataset. The goal is to classify whether a given text contains spoilers based on linguistic features and machine learning models.  

The workflow includes:  
- **Exploratory Data Analysis (EDA):** Understanding the structure and distribution of spoiler vs. non-spoiler text.  
- **Feature Engineering:** Transforming raw text data into meaningful numerical representations.  
- **Post-Feature Engineering EDA:** Evaluating the impact of feature transformations.  
- **Modeling:** Training baseline models using various machine learning approaches.  
- **Hyperparameter Tuning:** Optimizing model performance for improved spoiler detection.  

Here’s the README section for the data loading and preprocessing:  

## Data Loading and Preprocessing - `0_data_load.py`

The dataset consists of movie reviews and movie details sourced from the IMDB Spoiler Dataset. The following steps are performed:  

1. **Loading Data:**  
   - Reviews are loaded from `IMDB_reviews.json`.  
   - Movie details are loaded from `IMDB_movie_details.json`.  

2. **Preprocessing:**  
   - Duplicate reviews are removed, and the dataset is shuffled.  
   - `review_date` and `release_date` are converted to datetime format.  
   - `user_id` is encoded as categorical values.  

3. **Saving Processed Data:**  
   - The cleaned datasets are stored in `data/processed/reviews.csv` and `data/processed/movie_details.csv`.  

These steps ensure the data is structured and ready for exploratory analysis and feature engineering.  


## Exploratory Data Analysis (EDA) - `1_eda.py`

To understand the structure and characteristics of the dataset, an initial exploratory data analysis (EDA) was performed. Key analyses include:  

### **1. Data Overview**  
- Summary statistics and dataset shapes for reviews and movie details.  
- Data type conversions (e.g., datetime parsing, categorical encoding).  
- Handling missing values and duplicate entries.  

```bash
Reviews DataFrame Shape: (573385, 7)
Movies DataFrame Shape:  (1572, 8)   
```
```bash
Reviews DataFrame Info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 573385 entries, 0 to 573384
Data columns (total 7 columns):
 #   Column          Non-Null Count   Dtype            
---  ------          --------------   -----            
 0   review_date     573385 non-null  datetime64[ns]   
 1   movie_id        573385 non-null  object           
 2   user_id         573385 non-null  int64            
 3   is_spoiler      573385 non-null  bool             
 4   review_text     573385 non-null  object           
 5   rating          573385 non-null  int64            
 6   review_summary  573382 non-null  object           
dtypes: bool(1), datetime64[ns](1), int64(2), object(3)
memory usage: 26.8+ MB

Movies DataFrame Info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1572 entries, 0 to 1571
Data columns (total 8 columns):
 #   Column            Non-Null Count  Dtype              
---  ------            --------------  -----              
 0   movie_id          1572 non-null   object             
 1   plot_summary      1572 non-null   object             
 2   duration          1572 non-null   object             
 3   genre             1572 non-null   object             
 4   rating            1572 non-null   float64            
 5   release_date      1572 non-null   datetime64[ns]     
 6   plot_synopsis     1339 non-null   object             
 7   duration_minutes  1572 non-null   int64              
dtypes: datetime64[ns](1), float64(1), int64(1), object(5)
memory usage: 98.4+ KB
```
```bash
Missing Values in Reviews DataFrame:
review_date       0
movie_id          0
user_id           0
is_spoiler        0
review_text       0
rating            0
review_summary    3
dtype: int64

Missing Values in Movies DataFrame:
movie_id              0
plot_summary          0
duration              0
genre                 0
rating                0
release_date          0
plot_synopsis       233
duration_minutes      0
dtype: int64
```

### **2. Temporal Analysis**  
- **Review Trends:** Number of reviews over time, distribution by day of the week.  
- **Movie Releases:** Distribution of release years.  

![Reviews by Day of Week](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/initial/reviews_over_time.png)

![Movie Releases by Year](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/initial/releases_by_year.png)

### **3. Rating Distribution**  
- **Movie Ratings:** Histogram of movie ratings.  
- **Review Ratings:** Distribution of user review ratings.  

![Movie Rating Distribution](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/initial/ratings_distribution_by_movie.png)

![Review Rating Distribution](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/initial/ratings_distribution_by_review.png)

### **4. Genre Analysis**  
- **Top Movie Genres:** Frequency of genres.  
- **Average Rating by Genre:** Relationship between genre and movie ratings.  

![Average Rating by Genre](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/initial/avg_rating_by_genre.png)

![Movie Count by Genre](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/initial/movie_count_by_genre.png)

### **5. Review Text Analysis**  
- **Review Length Distribution:** Word count analysis.  
- **Review Length vs Rating:** Boxplot comparison.  

![Review Length Distribution](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/initial/review_length_distribution.png)

![Review Length by Rating](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/initial/review_length_by_rating.png)

### **6. Spoiler Analysis**  
- **Proportion of Spoilers:** Percentage of reviews marked as spoilers.  
- **Spoiler Rate by Rating:** How spoiler likelihood varies with review rating.  

![Spoiler Proportion](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/initial/spoiler_proportion.png)

![Spoiler Rate by Rating](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/initial/spoiler_rate_by_rating.png)

### **7. User Behavior Analysis**  
- **Review Counts per User:** Distribution of reviews per user.  
- **Top Reviewers:** Most active users by review count.  

![Review Counts per User](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/initial/user_review_count_distribution.png)

![Top 10 Most Active Users](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/initial/user_review_count_top10.png)

### **8. Movie Duration Analysis**  
- **Distribution of Durations:** Length of movies in minutes.  
- **Duration vs Rating:** Correlation between movie length and ratings. 

![Duration Distribution](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/initial/duration_distribution.png)

![Duration vs Rating](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/initial/duration_vs_rating.png)


### **Visualization Output**  
All plots are saved in the `eda/initial/` directory for further reference.  


## Feature Engineering - `2_feature_engineering.py`

To improve the predictive power of the model, multiple feature engineering techniques were applied to extract meaningful information from textual data, metadata, and timestamps.  

### **1. Text-Based Features**  
These features are derived from the textual content of reviews to quantify linguistic patterns and sentiment.  

- **Word and Sentence Statistics:**  
  - *Word Count:* The total number of words in a review. Longer reviews may provide more context but could also contain more spoilers.  
  - *Average Word Length:* The mean length of words in a review, which can indicate complexity.  
  - *Sentence Count:* The number of sentences in a review, helping to measure verbosity.  

- **Sentiment Scores (VADER Sentiment Analysis):**  
  - Uses the *Valence Aware Dictionary and sEntiment Reasoner (VADER)* algorithm to compute a *compound sentiment score*, which is a measure of overall sentiment ranging from -1 (negative) to +1 (positive).  
  - This helps determine if spoilers are associated with strong emotional reactions.  

- **Spoiler Keywords:**  
  - A predefined list of words associated with spoilers, such as *plot twist*, *reveals*, *ending*, and *betrayal*.  
  - A binary feature (`1` or `0`) is assigned based on whether each keyword appears in the review.  

- **Named Entity Recognition (NER):**  
  - Uses *natural language processing (NLP)* to identify specific types of named entities in text, such as:  
    - *PERSON* (character names)  
    - *LOCATION* (places mentioned in the movie)  
    - *ORGANIZATION* (e.g., studios, franchises)  
  - Counts of each entity type are included as features.  

- **Readability Metrics:**  
  - Calculated using the `textstat` library to measure how difficult a review is to read.  
  - **Flesch Reading Ease Score:** A higher score means easier readability.  
  - **Flesch-Kincaid Grade Level:** Estimates the U.S. school grade required to understand the text.  
  - **SMOG Index:** A readability formula that estimates comprehension difficulty based on polysyllabic words.  

### **2. Date Features**  
Extracting useful time-based insights from timestamps.  

- **Review Date Components:**  
  - Extracts *year*, *month*, and *day* from each review's timestamp to analyze seasonal trends in spoiler activity.  

- **Days Since Release:**  
  - The difference between the *review date* and the *movie release date* in days.  
  - Helps determine if reviews posted closer to the release date are more likely to contain spoilers.  

### **3. Movie Metadata Features**  
- **Movie Duration (Converted to Minutes):**  
  - Movie duration is typically stored as strings (e.g., `"2h 30m"`).  
  - A custom function converts this format into total minutes (e.g., `150` minutes).  

- **Aggregated Ratings:**  
  - Calculates summary statistics for review ratings on a per-movie basis:  
    - *Mean (Average)*: The average rating of a movie's reviews.  
    - *Median:* The middle rating, reducing the effect of outliers.  
    - *Standard Deviation:* Measures how spread out the ratings are, indicating rating consistency.  

### **4. Cosine Similarity Features**  
Cosine similarity is a metric that measures how similar two text documents are based on their word distributions.  

- **Review vs. Plot Summary Similarity:**  
  - Computes similarity between a review’s text and the movie’s plot summary using *TF-IDF vectorization* (explained below).  
  - Helps identify reviews that closely paraphrase official movie descriptions, which could indicate spoiler content.  

- **Review Summary vs. Synopsis Similarity:**  
  - Measures similarity between a user’s *summary* of a review and the full *synopsis* of the movie.  
  - Helps detect if short reviews summarize key movie events, making them potential spoilers.  

### **5. TF-IDF Features (Term Frequency-Inverse Document Frequency)**  
- *TF-IDF* is a numerical statistic that reflects how important a word is within a document relative to a larger collection of documents (the dataset).  
- The formula is:  
  \[
  \text{TF-IDF} = \text{Term Frequency (TF)} \times \text{Inverse Document Frequency (IDF)}
  \]  
  where:  
  - *TF* is the number of times a word appears in a review.  
  - *IDF* scales the importance by reducing the weight of common words (like *the*, *and*).  

- **TF-IDF with N-Grams:**  
  - Extracts single words (*unigrams*), two-word combinations (*bigrams*), and three-word combinations (*trigrams*).  
  - Captures contextual meaning, e.g., "plot twist" has a different meaning than "plot" and "twist" separately.  

- **Dimensionality Reduction with Truncated SVD:**  
  - *Singular Value Decomposition (SVD)* reduces the number of features while preserving important relationships.  
  - Helps avoid the "curse of dimensionality" and speeds up modeling.  

- **Hashing Vectorization:**  
  - A memory-efficient method that transforms text into fixed-length numerical representations without requiring a stored vocabulary.  
  - Useful for large datasets where traditional TF-IDF methods may be computationally expensive.  

### **6. Topic Modeling (Latent Dirichlet Allocation - LDA)**  
LDA is an *unsupervised learning* technique that discovers hidden topics in text.  

- **How it works:**  
  - Assumes each document (review) is a mixture of topics.  
  - Words in a document belong to different topics with different probabilities.  
  - For example, a review might contain **30% "romance" topic**, **40% "action" topic**, and **30% "spoiler" topic**.  

- **10 Topic Distributions:**  
  - Each review is assigned probabilities for **10 different latent topics**, extracted from the dataset.  
  - Helps the model learn what kind of topics are associated with spoilers.  

### **7. Statistical and Percentile Features**  
- **Review Length Percentile:**  
  - Ranks each review’s word count as a percentile within its movie's reviews.  
  - Helps determine if unusually short or long reviews are more likely to contain spoilers.  

### **8. Pipeline and Parallelization**  
- **Feature Extraction Pipelines:**  
  - Uses `sklearn.pipeline.Pipeline` to automate feature transformations.  
  - Ensures each processing step runs sequentially in an optimized way.  

- **Parallel Processing:**  
  - Computationally expensive operations (e.g., text analysis, TF-IDF transformations) are parallelized using `joblib.Parallel`.  
  - This speeds up processing by distributing work across multiple CPU cores.  

- **Logging:**  
  - All feature engineering steps are logged for monitoring and debugging.  
  - Logs are saved to `logs/feature_engineering_v3.log`.  

### **9. Data Storage**  
Processed datasets are saved in `data/processed/v3/` in **Parquet format**, which is:  
- **Efficient:** Compresses large datasets without losing information.  
- **Optimized for Queries:** Supports fast row-wise and column-wise retrieval.  

Files saved:  
- `reviews_engineered.parquet` – Processed review dataset with features.  
- `movies_engineered.parquet` – Processed movie dataset with metadata.  
- `merged.parquet` – Combined reviews and movie details.  
- `final_engineered.parquet` – Fully processed dataset ready for modeling.  



## Data Preprocessing and Splitting - `2.1_feature_engineering_splits.py`

Before training models, the dataset undergoes several preprocessing steps to ensure data quality, balance class distributions, and optimize feature representation.  

### **1. Handling Missing Values**  
- Missing values in textual fields (e.g., *review_summary*, *plot_synopsis*) are replaced with empty strings.  
- The *genre* column is treated as categorical and encoded numerically.  
- The target variable (*is_spoiler*) is converted to integer format for classification.  

### **2. Feature Selection**  
- Non-numeric and identifier columns (*user_id, review_text, movie_id, review_date, release_date*, etc.) are removed to retain only meaningful numerical features.  
- Any remaining object-type columns are encoded as categorical variables.  

### **3. Standardization**  
- Numerical features are standardized using *StandardScaler* to ensure uniform value ranges across all features.  
- The trained scaler is saved (`models/v3/prep/scaler.pkl`) to apply the same transformation during inference.  

### **4. Train-Test Splitting**  
- The dataset is split into **80% training data** and **20% testing data** using *stratified sampling* to maintain class distribution.  
- The baseline splits are saved under `data/processed/v3/splits/base/`.  

### **5. Handling Class Imbalance with SMOTE**  
- The dataset has an imbalance between spoiler and non-spoiler reviews.  
- *Synthetic Minority Over-sampling Technique (SMOTE)* is applied to generate synthetic samples for the minority class.  
- The resampled dataset is saved under `data/processed/v3/splits/smote/`.  

### **6. Dimensionality Reduction with PCA**  
- **Incremental Principal Component Analysis (PCA)** is used to reduce the number of features while retaining 95% of the variance.  
- The optimal number of components is determined based on cumulative explained variance.  
- The PCA-transformed dataset is saved under `data/processed/v3/splits/smote_pca/`.  

### **7. Logging and Metadata Storage**  
- Class distributions, processing steps, and feature names are logged for transparency and debugging.  
- Processing metadata (e.g., number of PCA components, explained variance) is stored in `models/v3/prep/preprocessing_metadata.pkl`.  

### **8. Visualization Outputs**  
- **Class Distribution Plot:** Shows the imbalance in spoiler vs. non-spoiler labels.  
- **PCA Explained Variance Plot:** Displays the variance retained as the number of components increases.  
- These figures are saved in `reports/v3/figures/`.  


## Post-Feature Engineering Exploratory Data Analysis (EDA) - `2.2_feature_engineered_eda.py`

After feature engineering, additional analysis was conducted to examine the impact of new features and their relationships with the target variable (*is_spoiler*). This step helps understand which features are most relevant for spoiler detection.  

### **1. Data Overview**  
- Summary of dataset size and number of features after processing.  
- Distribution of the target variable (*spoiler vs. non-spoiler reviews*).  
- Missing value analysis and visualization.  

```bash
Basic Information:
Total samples:  573,378
Total features: 180
```
```bash
Target variable distribution: is_spoiler
False    0.7369
True     0.2631
Name: proportion, dtype: float64
```
```bash
Missing Values Analysis:
                Missing Values  Missing Percent
plot_synopsis            35060         6.114640
review_summary               3         0.000523
```

### **2. Feature Distributions**  
- Histograms for numerical features to inspect their distributions.  
- Excluded high-dimensional features like SVD and LDA topics from basic analysis.  

![Review Length Distribution](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/engineered/numerical_dist_1.png)

![Review Rating Distribution](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/engineered/numerical_dist_7.png)

### **3. Correlation Analysis**  
- **Correlation with Target Variable:** Identifies which numerical features are most associated with spoilers.  
- **Top Positively & Negatively Correlated Features:** Highlights features with the strongest relationships to spoiler likelihood.  
- **Pairplot Analysis:** Visualizes interactions between top correlated features and spoilers.  

![Correlation with Target](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/engineered/target_correlation.png)

![Top Positively Correlated Features](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/engineered/top_pos_correlation.png)

![Top Negatively Correlated Features](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/engineered/top_neg_correlation.png)

### **4. Spoiler Keywords Analysis**  
- **Frequency Analysis:** Counts of spoiler-related keywords found in reviews.  
- **Keyword Correlation with Spoilers:** Identifies which keywords are most predictive of spoiler content.  

![Spoiler Keyword Frequency](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/engineered/spoiler_keyword_freq.png)

![Keyword Correlation with Spoilers](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/engineered/spoiler_keyword_corr.png)

### **5. Sentiment and Text Complexity Analysis**  
- **Sentiment Score by Spoiler Status:** Analyzes whether spoilers are associated with more extreme sentiment.  
- **Readability & Complexity Features:** Examines word count, sentence structure, and readability scores.  

![Sentiment Score by Spoiler Status](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/engineered/sentiment_by_spoiler.png)

![Readability & Complexity Features](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/engineered/text_complexity_corr.png)


### **6. Dimensionality Reduction Analysis**  
- **PCA on SVD Features:**  
  - Reduces SVD-transformed TF-IDF features to two principal components.  
  - Visualized in a 2D scatterplot to inspect separability of spoiler vs. non-spoiler reviews.  
- **Topic Modeling (LDA) Analysis:**  
  - Average topic weights compared between spoiler and non-spoiler reviews.  
  - Correlation of topics with spoiler likelihood.  

<!-- ![PCA on SVD Features](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/engineered/svd_pca_by_spoiler.png) -->

![Topic Modeling (LDA) Analysis](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/engineered/topic_avg_by_spoiler.png)

![Topic Correlation with Spoilers](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/engineered/topic_correlation.png)

### **7. Named Entity Recognition (NER) Analysis**  
- **Entity Frequency:** Counts of named entities (e.g., characters, locations) appearing in reviews.  
- **Entity Correlation with Spoilers:** Identifies whether certain named entities indicate spoiler content.  

<!-- ![Entity Frequency](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/engineered/entity_counts.png)

![Entity Correlation with Spoilers](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/engineered/entity_correlation.png)

![Entity Presence by Spoiler Status](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/engineered/entity_by_spoiler.png) -->

### **8. Text Similarity Analysis**  
- **Review vs. Plot Summary Similarity:** Measures if reviews closely match official plot descriptions.  
- **Review Summary vs. Synopsis Similarity:** Checks how well condensed reviews align with full movie synopses.  

![Review vs. Plot Summary Similarity](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/engineered/similarity_scores_by_spoiler.png)

![Review Summary vs. Synopsis Similarity](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/engineered/similarity_complexity_corr.png)

### **9. Temporal Analysis**  
- **Spoiler Rate Over Time:**  
  - Reviews analyzed by year and month to see if spoiler frequency changes over time.  
  - Days since movie release examined to determine if spoilers are more common soon after release.  

![Spoiler Rate Over Time](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/engineered/spoiler_by_year.png)

![Spoiler Rate by Movie Rating](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/engineered/spoiler_by_rating.png)

<!-- ![Days Since Release](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/engineered/spoiler_by_release_time.png) -->

### **10. Movie Metadata Analysis**  
- **Spoiler Frequency by Genre:** Determines if some genres are more prone to spoiler-heavy reviews.  
- **Spoiler Rate by Movie Rating:** Checks whether highly rated or poorly rated movies have more spoilers.  

![Spoiler Frequency by Genre](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/engineered/spoiler_by_genre.png)

### **11. Feature Importance Analysis**  
- **Top Features for Spoiler Prediction:**  
  - Identifies the most predictive features based on their correlation with *is_spoiler*.  
  - Results saved as visualizations for model interpretation.  

![Feature Importance](https://raw.githubusercontent.com/SantiagoEnriqueGA/movie_spoilers_nlp_dep/refs/heads/main/eda/engineered/feature_importance.png)

### **Visualization Output**  
All plots are saved in `eda/engineered/` for further analysis.  

