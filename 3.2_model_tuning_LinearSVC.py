from sklearn.svm import LinearSVC
from scipy.stats import loguniform, uniform
from model_tuning_utils import ModelTuner

# Define parameter distribution for Linear SVC
param_dist = {
    'C': loguniform(1e-4, 1e4),
    'loss': ['hinge', 'squared_hinge'],
    'penalty': ['l1', 'l2'],
    'dual': [False],  # False when using 'l1' penalty
    'tol': loguniform(1e-5, 1e-3),
    'class_weight': ['balanced', None],
    'intercept_scaling': uniform(0.1, 10),
    'max_iter': [2000]  # Increased max_iter to ensure convergence
}

# Initialize base model
base_model = LinearSVC(random_state=42)

# Initialize tuner
tuner = ModelTuner(
    model_name='Linear SVC',
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

print("Linear SVC tuning completed!")