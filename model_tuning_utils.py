import os
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

class ModelTuner:
    def __init__(
        self,
        model_name: str,
        base_model: Any,
        param_dist: Dict[str, Any],
        version: str = 'v3',
        n_iter: int = 100,
        cv: int = 5,
        n_jobs: int = -1,
        random_state: int = 42
    ):
        self.model_name = model_name
        self.base_model = base_model
        self.param_dist = param_dist
        self.version = version
        self.n_iter = n_iter
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.setup_logging()
        self.setup_directories()
        
    def setup_logging(self) -> None:
        """Set up logging configuration"""
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = f'{log_dir}/tuning_{self.model_name.lower().replace(" ", "_")}_{self.version}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f'model_tuning_{self.model_name.lower()}')
        
    def setup_directories(self) -> None:
        """Create necessary directories for storing results"""
        self.dirs = {
            'models': f'models/{self.version}/tuned',
            'reports': f'reports/{self.version}/tuning'
        }
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
            
    def load_data(self, data_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load training and test data"""
        data_path = f'data/processed/{self.version}/splits/{data_type.lower()}'
        self.logger.info(f"Loading data from {data_path}")
        
        try:
            X_train = joblib.load(f'{data_path}/X_train.pkl')
            X_test = joblib.load(f'{data_path}/X_test.pkl')
            y_train = joblib.load(f'{data_path}/y_train.pkl')
            y_test = joblib.load(f'{data_path}/y_test.pkl')
            
            self.logger.info(f"Training set shape: {X_train.shape}")
            self.logger.info(f"Test set shape: {X_test.shape}")
            self.logger.info(f"Class distribution in training: {np.bincount(y_train)}")
            
            return X_train, X_test, y_train, y_test
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
            
    def tune_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
        """Perform hyperparameter tuning using RandomizedSearchCV"""
        self.logger.info("Starting hyperparameter tuning")
        
        # Custom scoring metrics
        scoring = {
            'f1': make_scorer(f1_score),
            'roc_auc': make_scorer(roc_auc_score)
        }
        
        # Initialize RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=self.base_model,
            param_distributions=self.param_dist,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring=scoring,
            refit='f1',
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=1
        )
        
        # Perform search
        try:
            random_search.fit(X_train, y_train)
            self.logger.info(f"Best parameters: {random_search.best_params_}")
            self.logger.info(f"Best F1 score: {random_search.best_score_:.4f}")
            
            return random_search.best_estimator_, random_search.best_params_
        except Exception as e:
            self.logger.error(f"Error during tuning: {str(e)}")
            raise
            
    def evaluate_model(
        self,
        model: Any,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate the tuned model"""
        self.logger.info("Evaluating tuned model")
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=self.cv, scoring='f1')
        
        # Test set predictions
        y_pred = model.predict(X_test)
        test_f1 = f1_score(y_test, y_pred)
        
        # ROC AUC if model supports predict_proba
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
            test_roc_auc = roc_auc_score(y_test, y_prob)
        else:
            test_roc_auc = None
            
        results = {
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std(),
            'test_f1': test_f1,
            'test_roc_auc': test_roc_auc
        }
        
        self.logger.info(f"Evaluation results: {results}")
        return results
        
    def save_results(
        self,
        model: Any,
        best_params: Dict[str, Any],
        results: Dict[str, float],
        data_type: str
    ) -> None:
        """Save the tuned model and results"""
        # Save model
        model_path = f"{self.dirs['models']}/{self.model_name.lower().replace(' ', '_')}_{data_type.lower()}.pkl"
        joblib.dump(model, model_path)
        self.logger.info(f"Saved tuned model to {model_path}")
        
        # Save results
        results_dict = {
            'model_name': self.model_name,
            'data_type': data_type,
            'best_params': best_params,
            **results
        }
        
        results_path = f"{self.dirs['reports']}/{self.model_name.lower().replace(' ', '_')}_{data_type.lower()}.json"
        with open(results_path, 'w') as f:
            import json
            json.dump(results_dict, f, indent=4)
        self.logger.info(f"Saved results to {results_path}")
        
    def plot_learning_curves(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        data_type: str
    ) -> None:
        """Plot learning curves for the tuned model"""
        from sklearn.model_selection import learning_curve
        
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes, train_scores, valid_scores = learning_curve(
            model, X_train, y_train,
            train_sizes=train_sizes,
            cv=self.cv,
            scoring='f1',
            n_jobs=self.n_jobs
        )
        
        # Calculate means and stds
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        valid_mean = np.mean(valid_scores, axis=1)
        valid_std = np.std(valid_scores, axis=1)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='Training score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        plt.plot(train_sizes, valid_mean, label='Cross-validation score')
        plt.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, alpha=0.1)
        
        plt.xlabel('Training Examples')
        plt.ylabel('F1 Score')
        plt.title(f'Learning Curves - {self.model_name} ({data_type})')
        plt.legend(loc='best')
        plt.grid(True)
        
        # Save plot
        plt.savefig(f"{self.dirs['reports']}/learning_curve_{self.model_name.lower().replace(' ', '_')}_{data_type.lower()}.png")
        plt.close()