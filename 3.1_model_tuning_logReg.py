from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform, loguniform
from model_tuning_utils import ModelTuner

# Define parameter distribution for Logistic Regression
param_dist = {
    'C': loguniform(1e-4, 1e4),
    'penalty': ['l1', 'l2', 'elasticnet'],
    'solver': ['saga'],  # saga supports all penalties
    'l1_ratio': uniform(0, 1),  # for elasticnet
    'class_weight': ['balanced', None],
    'max_iter': [1000]
}

# Initialize base model
base_model = LogisticRegression(random_state=42)

# Initialize tuner
tuner = ModelTuner(
    model_name='Logistic Regression',
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
        # Load data
        X_train, X_test, y_train, y_test = tuner.load_data(data_type)
        
        # Tune model
        best_model, best_params = tuner.tune_model(X_train, y_train)
        
        # Evaluate model
        results = tuner.evaluate_model(best_model, X_train, X_test, y_train, y_test)
        
        # Save results
        tuner.save_results(best_model, best_params, results, data_type)
        
        # Plot learning curves
        tuner.plot_learning_curves(best_model, X_train, y_train, data_type)
        
    except Exception as e:
        tuner.logger.error(f"Error processing {data_type}: {str(e)}")
        continue

print("Logistic Regression tuning completed!")