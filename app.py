from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the model
model_path = r"D:\ML Financial Fraud Detection\models\xgboost_model.pkl"
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = []
    for i in range(1, 31):  # Adjust based on actual feature count
        feature_value = request.form.get(f'feature{i}')
        features.append(float(feature_value))
    
    features = np.array(features).reshape(1, -1)
    
    prediction = model.predict(features)
    proba = model.predict_proba(features)[:, 1]
    
    return render_template('index.html', prediction=int(prediction[0]), probability=float(proba[0]))

if __name__ == '__main__':
    app.run(debug=True)
