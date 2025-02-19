from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint, uniform
from model_tuning_utils import ModelTuner

# Define parameter distribution for AdaBoost
param_dist = {
    'n_estimators': randint(50, 300),
    'learning_rate': uniform(0.01, 1.99),  # 0.01 to 2
    'algorithm': ['SAMME', 'SAMME.R'],
    'base_estimator': [
        DecisionTreeClassifier(max_depth=1),
        DecisionTreeClassifier(max_depth=2),
        DecisionTreeClassifier(max_depth=3)
    ]
}

# Initialize base model
base_model = AdaBoostClassifier(random_state=42)

# Initialize tuner
tuner = ModelTuner(
    model_name='AdaBoost',
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

print("AdaBoost tuning completed!")