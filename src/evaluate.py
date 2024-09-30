import os
import logging
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler('evaluation.log'),
    logging.StreamHandler()
])

class ModelEvaluator:
    def __init__(self, model_paths, X_test, y_test):
        """Initialize the evaluator with model paths and test data."""
        self.model_paths = model_paths
        self.X_test = X_test
        self.y_test = y_test
        self.logger = logging.getLogger(__name__)

    def evaluate_model(self, model_path):
        """
        Load a model and evaluate its performance.
        """
        try:
            model = joblib.load(model_path)
            self.logger.info(f"Model loaded from {model_path}")

            # Make predictions
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)[:, 1]

            # Evaluate using accuracy, AUC-ROC, confusion matrix, etc.
            accuracy = accuracy_score(self.y_test, y_pred)
            auc_roc = roc_auc_score(self.y_test, y_proba)
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            class_report = classification_report(self.y_test, y_pred)

            # Log and print evaluation metrics
            self.logger.info(f"Accuracy: {accuracy:.4f}")
            self.logger.info(f"AUC-ROC: {auc_roc:.4f}")
            self.logger.info(f"Confusion Matrix:\n{conf_matrix}")
            self.logger.info(f"Classification Report:\n{class_report}")

            print(f"Accuracy: {accuracy:.4f}")
            print(f"AUC-ROC: {auc_roc:.4f}")
            print(f"Confusion Matrix:\n{conf_matrix}")
            print(f"Classification Report:\n{class_report}")

        except FileNotFoundError:
            self.logger.error(f"Model file not found: {model_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            raise

    def run_evaluation(self):
        """Run evaluation for all models."""
        for model_path in self.model_paths:
            self.evaluate_model(model_path)

if __name__ == "__main__":
    # Load the preprocessed test data from the pickle file
    try:
        with open(r"D:\ML Financial Fraud Detection\data\processed_data.pkl", 'rb') as f:
            X_train, X_test, y_train, y_test = pickle.load(f)

        model_paths = [
            r"D:\ML Financial Fraud Detection\models\logistic_regression.pkl",
            r"D:\ML Financial Fraud Detection\models\xgboost_model.pkl"
        ]

        evaluator = ModelEvaluator(model_paths, X_test, y_test)
        evaluator.run_evaluation()

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except Exception as e:
        logging.error(f"Error in loading data or running evaluation: {e}")
