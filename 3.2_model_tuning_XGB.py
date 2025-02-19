from xgboost import XGBClassifier
from scipy.stats import randint, uniform, loguniform
from model_tuning_utils import ModelTuner

# Define parameter distribution for XGBoost
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 10),
    'learning_rate': loguniform(1e-3, 1e-1),
    'subsample': uniform(0.6, 0.4),  # 0.6 to 1.0
    'colsample_bytree': uniform(0.6, 0.4),  # 0.6 to 1.0
    'min_child_weight': randint(1, 7),
    'gamma': uniform(0, 5),
    'reg_alpha': loguniform(1e-5, 1),
    'reg_lambda': loguniform(1e-5, 1),
    'scale_pos_weight': [1, 5, 10],  # For imbalanced datasets
    'max_delta_step': randint(0, 10)
}

# Initialize base model
base_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

# Initialize tuner
tuner = ModelTuner(
    model_name='XGBoost',
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

print("XGBoost tuning completed!")