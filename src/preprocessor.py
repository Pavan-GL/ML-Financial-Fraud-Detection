# src/preprocess.py

import pandas as pd
import logging
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter

# Configure logging
log_file_path = 'data_preprocessing.log'  # Log file path
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler(log_file_path),
    logging.StreamHandler()
])

class DataPreprocessor:
    def __init__(self, file_path):
        """
        Initializes the DataPreprocessor with the file path.
        
        Parameters:
        - file_path: str, path to the CSV file containing the dataset.
        """
        self.file_path = file_path
        self.output_dir = os.path.dirname(file_path)  # Use the same directory as the input file
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)

        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        """Load dataset from the specified file path."""
        try:
            self.logger.info(f"Loading data from {self.file_path}")
            data = pd.read_csv(self.file_path)
            self.logger.info(f"Data loaded successfully. Shape: {data.shape}")
            return data
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {e}")
            raise
        except pd.errors.EmptyDataError as e:
            self.logger.error("No data: The file is empty.")
            raise

    def preprocess_data(self, data):
        """Scale features and split the dataset into train and test sets."""
        try:
            data['scaled_amount'] = self.scaler.fit_transform(data[['Amount']])
            data['scaled_time'] = self.scaler.fit_transform(data[['Time']])
            data.drop(['Amount', 'Time'], axis=1, inplace=True)
            self.logger.info("Feature scaling completed and original columns dropped.")

            X = data.drop('Class', axis=1)
            y = data['Class']

            # Log class distribution
            class_distribution = Counter(y)
            self.logger.info(f"Class distribution: {class_distribution}")

            # Check for class imbalance
            if min(class_distribution.values()) < 2:
                self.logger.warning("One or more classes have fewer than 2 members. Removing these classes.")
                y = y[y.map(class_distribution) > 1]
                X = X.loc[y.index]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            self.logger.info("Data split into train and test sets.")

            return X_train, X_test, y_train, y_test
        except Exception as e:
            self.logger.error(f"An error occurred during preprocessing: {e}")
            raise

    def save_to_pickle(self, data):
        """Save processed data to a pickle file."""
        output_file = os.path.join(self.output_dir, 'processed_data.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        self.logger.info(f"Processed data saved to {output_file}")

    def run(self):
        """Run the complete data loading and preprocessing pipeline."""
        data = self.load_data()
        X_train, X_test, y_train, y_test = self.preprocess_data(data)
        self.save_to_pickle((X_train, X_test, y_train, y_test))
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Replace the path below with your actual CSV file path
    file_path = "D:\ML Financial Fraud Detection\data\creditcard.csv" # Update with your actual CSV file path
    preprocessor = DataPreprocessor(file_path)
    X_train, X_test, y_train, y_test = preprocessor.run()
