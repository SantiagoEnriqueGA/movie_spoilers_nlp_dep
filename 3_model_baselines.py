import os
import logging
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    VotingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from sklearn.exceptions import ConvergenceWarning
import warnings

# Suppress common warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Version Indicator and configuration
VERSION = 'v3'  # Updated version
DATA_TYPES = ['BASE', 'SMOTE', 'SMOTE_PCA']  # Process all data types
SAVE_PLOTS = True
RETRAIN_MODELS = True

# Set up logging
os.makedirs('logs', exist_ok=True)
os.makedirs(f'reports/{VERSION}/figures', exist_ok=True)

# Create a function to train and evaluate models for each data type
def train_and_evaluate(data_type):
    # Set up logging for this data type
    log_file = f'logs/model_training_{data_type.lower()}_{VERSION}.log'
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file,
        filemode='w'
    )
    logger = logging.getLogger(f'model_training_{data_type.lower()}')
    
    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    logger.info(f"Starting model training for {data_type} data")
    
    # Load the appropriate data
    data_path = f'data/processed/{VERSION}/splits/{data_type.lower()}'
    try:
        X_train = joblib.load(f'{data_path}/X_train.pkl')
        X_test = joblib.load(f'{data_path}/X_test.pkl')
        y_train = joblib.load(f'{data_path}/y_train.pkl')
        y_test = joblib.load(f'{data_path}/y_test.pkl')
        logger.info(f"Successfully loaded data from {data_path}")
        logger.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        logger.info(f"Training label distribution: {np.bincount(y_train)}")
        logger.info(f"Testing label distribution: {np.bincount(y_test)}")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return
    
    # Models to train with hyperparameters optimized for imbalanced data
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, 
            random_state=42,
            class_weight='balanced',
            C=0.1,  # Regularization parameter
            n_jobs=-1
        ),
        'SGD Classifier': SGDClassifier(
            random_state=42,
            loss='modified_huber',
            class_weight='balanced',
            n_jobs=-1
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            random_state=42,
            # Scale the positive weights to balance the classes
            scale_pos_weight=np.bincount(y_train)[0]/np.bincount(y_train)[1] if len(np.bincount(y_train)) > 1 else 1,
            n_jobs=-1
        ),
        'LightGBM': LGBMClassifier(
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            random_state=42,
            n_estimators=100,
            subsample=0.8,
            max_depth=5
        ),
        'AdaBoost': AdaBoostClassifier(
            random_state=42,
            n_estimators=100
        ),
        'Linear SVC': LinearSVC(
            random_state=42,
            class_weight='balanced',
            max_iter=2000,
            dual=False
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=42,
            class_weight='balanced',
            max_depth=10
        )
    }
    
    # Create directory for saving models
    os.makedirs(f'models/{VERSION}/{data_type.lower()}', exist_ok=True)
    os.makedirs(f'reports/{VERSION}/{data_type.lower()}', exist_ok=True)
    
    # Results storage for comparison
    results = {
        'Model': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': [],
        'AUC-ROC': [],
        'Training Time': []
    }
    
    # Train and evaluate each model
    for model_name, model in tqdm(models.items(), desc=f"Training models on {data_type} data"):
        logger.info(f'Starting training for {model_name}...')
        
        # Skip retraining if model exists and RETRAIN_MODELS is False
        model_path = f'models/{VERSION}/{data_type.lower()}/{model_name.replace(" ", "_").lower()}_model.pkl'
        if not RETRAIN_MODELS and os.path.exists(model_path):
            logger.info(f'Loading existing {model_name} model')
            model = joblib.load(model_path)
        else:
            try:
                # Time the training
                start_time = time.time()
                
                # Special handling for SVM which might not converge
                if model_name == 'Linear SVC':
                    try:
                        model.fit(X_train, y_train)
                    except Exception as e:
                        logger.warning(f"SVC training failed with error: {str(e)}. Trying with increased max_iter.")
                        model = LinearSVC(random_state=42, class_weight='balanced', max_iter=5000, dual=False)
                        model.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train)
                
                training_time = time.time() - start_time
                logger.info(f'{model_name} trained in {training_time:.2f} seconds')
                
                # Save the trained model
                joblib.dump(model, model_path)
                logger.info(f'Saved {model_name} model to {model_path}')
                
            except Exception as e:
                logger.error(f'Error training {model_name}: {str(e)}')
                continue
        
        # Evaluate the model
        try:
            # Get predictions
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_prob = model.decision_function(X_test)
                # Normalize to [0,1] for easier interpretation
                y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
            else:
                y_prob = None
            
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Store results
            results['Model'].append(model_name)
            results['Accuracy'].append(report['accuracy'])
            results['Precision'].append(report['1']['precision'])
            results['Recall'].append(report['1']['recall'])
            results['F1-Score'].append(report['1']['f1-score'])
            results['Training Time'].append(training_time if RETRAIN_MODELS else 'N/A')
            
            # Calculate ROC AUC if probabilities are available
            if y_prob is not None:
                auc_score = roc_auc_score(y_test, y_prob)
                results['AUC-ROC'].append(auc_score)
                logger.info(f'ROC AUC for {model_name}: {auc_score:.4f}')
                
                # Plot ROC curve if requested
                if SAVE_PLOTS:
                    plt.figure(figsize=(10, 8))
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {auc_score:.4f})')
                    plt.plot([0, 1], [0, 1], 'k--', lw=2)
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'ROC Curve - {model_name} ({data_type})')
                    plt.legend(loc="lower right")
                    plt.savefig(f'reports/{VERSION}/{data_type.lower()}/roc_{model_name.replace(" ", "_").lower()}.png')
                    plt.close()
                    
                    # Precision-Recall curve
                    precision, recall, _ = precision_recall_curve(y_test, y_prob)
                    avg_precision = average_precision_score(y_test, y_prob)
                    plt.figure(figsize=(10, 8))
                    plt.plot(recall, precision, lw=2, label=f'{model_name} (AP = {avg_precision:.4f})')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    plt.title(f'Precision-Recall Curve - {model_name} ({data_type})')
                    plt.legend(loc="lower left")
                    plt.savefig(f'reports/{VERSION}/{data_type.lower()}/pr_{model_name.replace(" ", "_").lower()}.png')
                    plt.close()
            else:
                results['AUC-ROC'].append('N/A')
            
            # Log classification report and confusion matrix
            logger.info(f'Classification Report for {model_name}:')
            logger.info(f'\n{classification_report(y_test, y_pred)}')
            logger.info(f'Confusion Matrix for {model_name}:')
            logger.info(f'\n{conf_matrix}')
            
            # Save confusion matrix plot
            if SAVE_PLOTS:
                plt.figure(figsize=(8, 6))
                plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title(f'Confusion Matrix - {model_name} ({data_type})')
                plt.colorbar()
                classes = ['Not Spoiler', 'Spoiler']
                tick_marks = np.arange(len(classes))
                plt.xticks(tick_marks, classes, rotation=45)
                plt.yticks(tick_marks, classes)
                
                # Add text annotations
                thresh = conf_matrix.max() / 2.0
                for i in range(conf_matrix.shape[0]):
                    for j in range(conf_matrix.shape[1]):
                        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                                horizontalalignment="center",
                                color="white" if conf_matrix[i, j] > thresh else "black")
                
                plt.tight_layout()
                plt.ylabel('True label')
                plt.xlabel('Predicted label')
                plt.savefig(f'reports/{VERSION}/{data_type.lower()}/cm_{model_name.replace(" ", "_").lower()}.png')
                plt.close()
            
        except Exception as e:
            logger.error(f'Error evaluating {model_name}: {str(e)}')
            continue
    
    # Create ensemble model from best performers
    try:
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'reports/{VERSION}/{data_type.lower()}/model_comparison.csv', index=False)
        
        # Plot model comparison
        if SAVE_PLOTS:
            # Accuracy
            plt.figure(figsize=(14, 10))
            results_df.sort_values('Accuracy', ascending=False, inplace=True)
            plt.barh(results_df['Model'], results_df['Accuracy'])
            plt.title(f'Model Accuracy Comparison ({data_type})')
            plt.xlabel('Accuracy')
            plt.tight_layout()
            plt.savefig(f'reports/{VERSION}/{data_type.lower()}/accuracy_comparison.png')
            plt.close()
            
            # F1 Score
            plt.figure(figsize=(14, 10))
            results_df.sort_values('F1-Score', ascending=False, inplace=True)
            plt.barh(results_df['Model'], results_df['F1-Score'])
            plt.title(f'Model F1-Score Comparison ({data_type})')
            plt.xlabel('F1-Score')
            plt.tight_layout()
            plt.savefig(f'reports/{VERSION}/{data_type.lower()}/f1_comparison.png')
            plt.close()
        
        # Create ensemble of top models if we retrained models
        if RETRAIN_MODELS:
            # Select top 3 models based on F1-Score
            top_models = results_df.nlargest(3, 'F1-Score')
            logger.info(f"Creating ensemble from top models: {top_models['Model'].tolist()}")
            
            estimators = []
            for model_name in top_models['Model']:
                model_path = f'models/{VERSION}/{data_type.lower()}/{model_name.replace(" ", "_").lower()}_model.pkl'
                loaded_model = joblib.load(model_path)
                estimators.append((model_name.replace(" ", "_").lower(), loaded_model))
            
            # Create and train voting classifier
            ensemble = VotingClassifier(estimators=estimators, voting='soft')
            ensemble.fit(X_train, y_train)
            
            # Save ensemble model
            joblib.dump(ensemble, f'models/{VERSION}/{data_type.lower()}/ensemble_model.pkl')
            logger.info(f"Saved ensemble model to models/{VERSION}/{data_type.lower()}/ensemble_model.pkl")
            
            # Evaluate ensemble
            if all(hasattr(m, 'predict_proba') for _, m in estimators):
                y_prob_ensemble = ensemble.predict_proba(X_test)[:, 1]
                auc_score_ensemble = roc_auc_score(y_test, y_prob_ensemble)
                logger.info(f'Ensemble ROC AUC: {auc_score_ensemble:.4f}')
            
            y_pred_ensemble = ensemble.predict(X_test)
            report_ensemble = classification_report(y_test, y_pred_ensemble)
            logger.info(f'Ensemble Classification Report:')
            logger.info(f'\n{report_ensemble}')
        
    except Exception as e:
        logger.error(f'Error creating ensemble model: {str(e)}')
    
    logger.info(f"Completed model training for {data_type} data")

# Process each data type
for data_type in DATA_TYPES:
    train_and_evaluate(data_type)

# Final comparison across data types
try:
    all_results = []
    for data_type in DATA_TYPES:
        results_path = f'reports/{VERSION}/{data_type.lower()}/model_comparison.csv'
        if os.path.exists(results_path):
            df = pd.read_csv(results_path)
            df['Data Type'] = data_type
            all_results.append(df)
    
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_results.to_csv(f'reports/{VERSION}/all_model_comparison.csv', index=False)
        
        # Plot best models by data type
        plt.figure(figsize=(16, 12))
        for i, data_type in enumerate(DATA_TYPES):
            data = combined_results[combined_results['Data Type'] == data_type]
            if not data.empty:
                best_model = data.nlargest(1, 'F1-Score')
                plt.bar(i, best_model['F1-Score'].values[0], label=f"{data_type}: {best_model['Model'].values[0]}")
        
        plt.title('Best Model F1-Score by Data Type')
        plt.xticks(range(len(DATA_TYPES)), DATA_TYPES)
        plt.xlabel('Data Type')
        plt.ylabel('F1-Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'reports/{VERSION}/best_models_comparison.png')
        plt.close()
        
except Exception as e:
    print(f'Error creating final comparison: {str(e)}')

print("Model training and evaluation completed for all data types!")