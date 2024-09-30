import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of rows (1000 transactions)
n_rows = 1000

# Generate features
time = np.random.uniform(0, 172792, size=n_rows)  # Random time of transaction
amount = np.random.uniform(0, 5000, size=n_rows)  # Random transaction amount between 0 and 5000

# Create 28 anonymized 'V' features (as in the original dataset)
V_columns = [f'V{i}' for i in range(1, 29)]
V_data = np.random.normal(size=(n_rows, 28))

# Generate target variable (Class: 0 for non-fraud, 1 for fraud)
fraud_probability = 0.00172  # Fraud class probability of ~0.172%
fraud_class = np.random.choice([0, 1], size=n_rows, p=[1 - fraud_probability, fraud_probability])

# Rebalance the dataset to ensure we have both fraudulent and non-fraudulent cases

# Set the number of fraudulent transactions based on the 0.172% fraud ratio
n_fraud = int(0.00172 * n_rows) if int(0.00172 * n_rows) > 0 else 1  # Ensure at least 1 fraudulent transaction

# Create a balanced dataset
fraud_indices = np.random.choice(n_rows, n_fraud, replace=False)  # Randomly select fraud indices



# Combine all features into a DataFrame
data = pd.DataFrame(V_data, columns=V_columns)
data['Time'] = time
data['Amount'] = amount
data['Class'] = fraud_class

# Rebalance the dataset to ensure we have both fraudulent and non-fraudulent cases

n_fraud = 100

# Create a balanced dataset
fraud_indices = np.random.choice(n_rows, n_fraud, replace=False)  # Randomly select fraud indices
data['Class'] = 0  # Set all transactions to non-fraudulent by default
data.loc[fraud_indices, 'Class'] = 1

# Ensure there is at least one fraud
data['Class'].value_counts()

# Display the dataset to verify
data.head()


# Show the generated dataset
print(data.head())
data.to_csv('creditcard.csv',header=True,index=False)
