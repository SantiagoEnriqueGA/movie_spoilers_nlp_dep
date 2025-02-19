from sklearn.linear_model import SGDClassifier
from scipy.stats import uniform, loguniform
from model_tuning_utils import ModelTuner

# Define parameter distribution for SGD Classifier
param_dist = {
    'loss': ['log_loss', 'modified_huber', 'perceptron'],
    'penalty': ['l1', 'l2', 'elasticnet'],
    'alpha': loguniform(1e-6, 1e-2),
    'l1_ratio': uniform(0, 1),
    'learning_rate': ['optimal', 'adaptive'],
    'class_weight': ['balanced', None],
    'epsilon': uniform(0.01, 0.1),  # for modified_huber loss
    'max_iter': [1000],
    'eta0': uniform(0.01, 1.0),  # initial learning rate
    'power_t': uniform(0.1, 0.9)  # power for inverse scaling learning rate
}

# Initialize base model
base_model = SGDClassifier(random_state=42)

# Initialize tuner
tuner = ModelTuner(
    model_name='SGD Classifier',
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

print("SGD Classifier tuning completed!")