from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import randint, uniform, loguniform
from model_tuning_utils import ModelTuner

# Define parameter distribution for Gradient Boosting
param_dist = {
    'n_estimators': randint(100, 500),
    'learning_rate': loguniform(1e-3, 1e-1),
    'max_depth': randint(3, 10),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'subsample': uniform(0.6, 0.4),  # 0.6 to 1.0
    'max_features': ['sqrt', 'log2', None],
    'criterion': ['friedman_mse', 'squared_error'],
    'validation_fraction': uniform(0.1, 0.2),
    'n_iter_no_change': randint(5, 20),
    'tol': loguniform(1e-5, 1e-3)
}

# Initialize base model
base_model = GradientBoostingClassifier(random_state=42)

# Initialize tuner
tuner = ModelTuner(
    model_name='Gradient Boosting',
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

print("Gradient Boosting tuning completed!")