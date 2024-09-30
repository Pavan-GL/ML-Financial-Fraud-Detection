import os
import logging
import joblib
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler('model_training.log'),
    logging.StreamHandler()
])

class ModelTrainer:
    def __init__(self):
        """Initializes the ModelTrainer, setting the path for the preprocessed data."""
        self.pickle_file = r"D:\ML Financial Fraud Detection\data\processed_data.pkl"  # Path for the preprocessed data
        self.model_dir = r"..\models"  # Directory for saving models
        self.logger = logging.getLogger(__name__)
        os.makedirs(self.model_dir, exist_ok=True)  # Create model directory if it doesn't exist

    def load_data(self):
        """Load preprocessed data from the pickle file."""
        try:
            with open(self.pickle_file, 'rb') as f:
                X_train, X_test, y_train, y_test = pickle.load(f)
            self.logger.info("Data loaded from pickle file successfully.")
            return X_train, X_test, y_train, y_test
        except FileNotFoundError as e:
            self.logger.error(f"Pickle file not found: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading data from pickle file: {e}")
            raise

    def check_class_distribution(self, y_train):
        """Check the class distribution in the training data."""
        unique, counts = np.unique(y_train, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        self.logger.info(f"Class distribution in training data: {class_distribution}")
        self.logger.info(f"Unique classes in y_train: {unique}")
        self.logger.info(f"Counts in y_train: {counts}")

        if len(class_distribution) < 2:
            self.logger.error("Training data contains only one class.")
            raise ValueError("Training data must contain at least two classes.")

    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression model and save it as a pickle file."""
        self.check_class_distribution(y_train)  # Check class distribution
        try:
            log_model = LogisticRegression(max_iter=1000)
            log_model.fit(X_train, y_train)
            model_path = os.path.join(self.model_dir, 'logistic_regression.pkl')
            joblib.dump(log_model, model_path)  # Save the model to a pickle file
            self.logger.info(f"Logistic Regression model trained and saved to {model_path}.")
        except Exception as e:
            self.logger.error(f"Error during Logistic Regression model training: {e}")
            raise

    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model and save it as a pickle file."""
        self.check_class_distribution(y_train)  # Check class distribution
        try:
            xgb_model = XGBClassifier()
            xgb_model.fit(X_train, y_train)
            model_path = os.path.join(self.model_dir, 'xgboost_model.pkl')
            joblib.dump(xgb_model, model_path)  # Save the model to a pickle file
            self.logger.info(f"XGBoost model trained and saved to {model_path}.")
        except Exception as e:
            self.logger.error(f"Error during XGBoost model training: {e}")
            raise

    def run(self):
        """Run the training process."""
        X_train, X_test, y_train, y_test = self.load_data()
        self.train_logistic_regression(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        self.logger.info("All models trained and saved successfully.")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()
