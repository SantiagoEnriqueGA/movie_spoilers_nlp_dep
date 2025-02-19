from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform, loguniform
from model_tuning_utils import ModelTuner

# Define parameter distribution for LightGBM
param_dist = {
    'n_estimators': randint(100, 500),
    'learning_rate': loguniform(1e-3, 1e-1),
    'num_leaves': randint(20, 100),
    'max_depth': randint(3, 10),
    'min_child_samples': randint(10, 50),
    'subsample': uniform(0.6, 0.4),  # 0.6 to 1.0
    'colsample_bytree': uniform(0.6, 0.4),  # 0.6 to 1.0
    'reg_alpha': loguniform(1e-5, 1),
    'reg_lambda': loguniform(1e-5, 1),
    'min_split_gain': uniform(0, 1),
    'class_weight': ['balanced', None],
    'boosting_type': ['gbdt', 'dart']
}

# Initialize base model
base_model = LGBMClassifier(random_state=42)

# Initialize tuner
tuner = ModelTuner(
    model_name='LightGBM',
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

print("LightGBM tuning completed!")