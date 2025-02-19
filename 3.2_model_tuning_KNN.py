from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import randint
from model_tuning_utils import ModelTuner

# Define parameter distribution for KNN
param_dist = {
    'n_neighbors': randint(3, 20),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': randint(20, 100),
    'p': [1, 2],  # Manhattan or Euclidean distance
    'metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev']
}

# Initialize base model
base_model = KNeighborsClassifier()

# Initialize tuner
tuner = ModelTuner(
    model_name='K-Nearest Neighbors',
    base_model=base_model,
    param_dist=param_dist,
    n_iter=100,
    cv=5,
    n_jobs=-1,
)

# Data types to process
DATA_TYPES = ['BASE', 'SMOTE', 'SMOTE_PCA']

# Tune model for each data type
for data_type in DATA_TYPES:
    try:
        X_train, X_test, y_train, y_test = tuner.load_data(data_type)
        best_model, best_params = tuner.tune_model(X_train, y_train)
        results = tuner.evaluate_model(best_model, X_train, X_test, y_train, y_test)
        tuner.save_results(best_model, best_params, results, data_type)
        tuner.plot_learning_curves(best_model, X_train, y_train, data_type)
    except Exception as e:
        tuner.logger.error(f"Error processing {data_type}: {str(e)}")
        continue

print("KNN tuning completed!")