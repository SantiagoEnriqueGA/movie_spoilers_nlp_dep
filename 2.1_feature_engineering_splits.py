import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix, vstack
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import logging
import warnings

# Version Indicator
VERSION = 'v3'

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=f'logs/feature_engineering_splits_{VERSION}.log',
    filemode='w'
    )
logger = logging.getLogger('feature_engineering')

# Create necessary directories
os.makedirs(f'data/processed/{VERSION}/splits/base', exist_ok=True)
os.makedirs(f'data/processed/{VERSION}/splits/smote', exist_ok=True)
os.makedirs(f'data/processed/{VERSION}/splits/smote_pca', exist_ok=True)
os.makedirs(f'models/{VERSION}/prep', exist_ok=True)
os.makedirs(f'reports/{VERSION}/figures', exist_ok=True)

# Load the data
try:
    df = pd.read_parquet(f'data/processed/v3/final_engineered.parquet')  # Using previous version's data
    logger.info(f"Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
except FileNotFoundError:
    logger.error("Dataset file not found. Please check the file path.")
    raise

# Display column data types
for col in df.columns:
    logger.debug(f"{col}: {df[col].dtype}")

# Data preprocessing
logger.info("Starting data preprocessing...")

# Handle missing values
df['review_summary'] = df['review_summary'].fillna('')
df['plot_synopsis'] = df['plot_synopsis'].fillna('')
df['plot_summary'] = df['plot_summary'].fillna('')
df['genre'] = df['genre'].fillna('Unknown')
df['genre'] = df['genre'].astype('category').cat.codes

# Ensure target variable is properly formatted
df['is_spoiler'] = df['is_spoiler'].astype(int)

# Select features - dropping text-based and identifier columns
X = df.drop(columns=[
    'user_id', 'is_spoiler', 'review_text', 'plot_summary', 
    'plot_synopsis', 'movie_id', 'review_date', 'release_date',
    'review_summary', 'duration' # Dropped additional non-numeric columns
])

# Handle any remaining non-numeric columns 
for col in X.columns:
    if X[col].dtype == 'object':
        logger.warning(f"Converting object column {col} to categorical")
        X[col] = X[col].astype('category').cat.codes

y = df['is_spoiler']

# Log class distribution
logger.info("Initial target class distribution:")
logger.info(f"{y.value_counts()}")
logger.info(f"Class imbalance ratio: 1:{y.value_counts()[0]/y.value_counts()[1]:.2f}")

# Plot class distribution
plt.figure(figsize=(10, 6))
y.value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Class Distribution')
plt.xlabel('Is Spoiler')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Not Spoiler', 'Spoiler'], rotation=0)
plt.tight_layout()
plt.savefig(f'reports/{VERSION}/figures/class_distribution.png')
plt.close()

# Standardize numerical features
logger.info("Standardizing numerical features...")
feature_names = X.columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, f'models/{VERSION}/prep/scaler.pkl')
logger.info(f"Saved scaler to models/{VERSION}/prep/scaler.pkl")

# Split the data with stratification to maintain class distribution
logger.info("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

logger.info("Training set target class counts:")
logger.info(f"{pd.Series(y_train).value_counts()}")
logger.info("Testing set target class counts:")
logger.info(f"{pd.Series(y_test).value_counts()}")

# Save baseline splits
# ---------------------------------------------------------------------------------------------
logger.info("Saving baseline splits...")
joblib.dump(X_train, f'data/processed/{VERSION}/splits/base/X_train.pkl')
joblib.dump(X_test, f'data/processed/{VERSION}/splits/base/X_test.pkl')
joblib.dump(y_train, f'data/processed/{VERSION}/splits/base/y_train.pkl')
joblib.dump(y_test, f'data/processed/{VERSION}/splits/base/y_test.pkl')

# Apply SMOTE for handling class imbalance
# ---------------------------------------------------------------------------------------------
logger.info("Applying SMOTE to handle class imbalance...")
# Use 5 for k_neighbors if minority class has fewer samples, otherwise use default
minority_sample_count = min(pd.Series(y_train).value_counts())
k_neighbors = min(5, minority_sample_count-1) if minority_sample_count < 6 else 5

try:
    nn = NearestNeighbors(n_neighbors=k_neighbors, n_jobs=-1)
    smote = SMOTE(random_state=42, k_neighbors=nn)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    logger.info("SMOTE completed successfully")
except ValueError as e:
    logger.error(f"SMOTE failed: {str(e)}")
    logger.warning("Falling back to original data due to SMOTE failure")
    X_train_smote, y_train_smote = X_train, y_train

logger.info("SMOTE-resampled training set target class counts:")
logger.info(f"{pd.Series(y_train_smote).value_counts()}")

# Save SMOTE-resampled data
logger.info("Saving SMOTE-resampled splits...")
joblib.dump(X_train_smote, f'data/processed/{VERSION}/splits/smote/X_train.pkl')
joblib.dump(X_test, f'data/processed/{VERSION}/splits/smote/X_test.pkl')
joblib.dump(y_train_smote, f'data/processed/{VERSION}/splits/smote/y_train.pkl')
joblib.dump(y_test, f'data/processed/{VERSION}/splits/smote/y_test.pkl')

# Apply PCA to reduce dimensionality while preserving information
# ---------------------------------------------------------------------------------------------
logger.info("Applying Incremental PCA for dimensionality reduction...")

# First determine optimal number of components
logger.info("Determining optimal number of components...")
batch_size = min(1000, X_train_smote.shape[0])
initial_pca = IncrementalPCA(n_components=min(125, X_train_smote.shape[1]), batch_size=batch_size)

# Process in batches to avoid memory issues
for i in range(0, X_train_smote.shape[0], batch_size):
    end = min(i + batch_size, X_train_smote.shape[0])
    initial_pca.partial_fit(X_train_smote[i:end])

# Find number of components for 95% variance
cumulative_variance_ratio = np.cumsum(initial_pca.explained_variance_ratio_)
n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
logger.info(f"Using {n_components_95} components to explain 95% of variance")

# Plot explained variance
plt.figure(figsize=(12, 6))
plt.plot(cumulative_variance_ratio, marker='o', linestyle='--', color='b')
plt.axhline(y=0.95, color='r', linestyle='-')
plt.axvline(x=n_components_95, color='g', linestyle='-')
plt.title('Cumulative Explained Variance vs. Number of Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.savefig(f'reports/{VERSION}/figures/pca_explained_variance.png')
plt.close()

# Apply final PCA with optimal components
final_pca = IncrementalPCA(n_components=n_components_95, batch_size=batch_size)

# Fit PCA in batches
for i in range(0, X_train_smote.shape[0], batch_size):
    end = min(i + batch_size, X_train_smote.shape[0])
    final_pca.partial_fit(X_train_smote[i:end])

# Save the PCA model
joblib.dump(final_pca, f'models/{VERSION}/prep/pca.pkl')
logger.info(f"Saved PCA model to models/{VERSION}/prep/pca.pkl")

# Transform the data in batches
logger.info("Transforming data with PCA...")
X_train_smote_pca = np.empty((X_train_smote.shape[0], n_components_95))
X_test_pca = np.empty((X_test.shape[0], n_components_95))

for i in range(0, X_train_smote.shape[0], batch_size):
    end = min(i + batch_size, X_train_smote.shape[0])
    X_train_smote_pca[i:end] = final_pca.transform(X_train_smote[i:end])

for i in range(0, X_test.shape[0], batch_size):
    end = min(i + batch_size, X_test.shape[0])
    X_test_pca[i:end] = final_pca.transform(X_test[i:end])

# Convert to CSR matrix for efficiency if sparse data needed
X_train_smote_pca_sparse = csr_matrix(X_train_smote_pca)
X_test_pca_sparse = csr_matrix(X_test_pca)

# Save the PCA-transformed data
logger.info("Saving PCA-transformed splits...")
joblib.dump(X_train_smote_pca_sparse, f'data/processed/{VERSION}/splits/smote_pca/X_train.pkl')
joblib.dump(X_test_pca_sparse, f'data/processed/{VERSION}/splits/smote_pca/X_test.pkl')
joblib.dump(y_train_smote, f'data/processed/{VERSION}/splits/smote_pca/y_train.pkl')
joblib.dump(y_test, f'data/processed/{VERSION}/splits/smote_pca/y_test.pkl')

# Save feature names for interpretation
# ---------------------------------------------------------------------------------------------
with open(f'models/{VERSION}/prep/feature_names.txt', 'w') as f:
    for feature in feature_names:
        f.write(f"{feature}\n")

logger.info("Data preprocessing and transformation completed successfully!")

# Save important processing metadata
metadata = {
    'original_shape': df.shape,
    'processed_features': len(feature_names),
    'pca_components': n_components_95,
    'explained_variance': cumulative_variance_ratio[n_components_95-1],
    'class_distribution_original': y.value_counts().to_dict(),
    'class_distribution_train': pd.Series(y_train).value_counts().to_dict(),
    'class_distribution_train_smote': pd.Series(y_train_smote).value_counts().to_dict(),
    'class_distribution_test': pd.Series(y_test).value_counts().to_dict(),
}

joblib.dump(metadata, f'models/{VERSION}/prep/preprocessing_metadata.pkl')
logger.info(f"Saved preprocessing metadata to models/{VERSION}/prep/preprocessing_metadata.pkl")

logger.info("All processing completed successfully!")