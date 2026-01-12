"""
Make predictions using trained XGBoost model

This script loads a trained model and makes predictions on test data.

Input:  models/xgboost_model.pkl, data/processed/X_test, y_test
Output: Predictions and evaluation metrics

Usage:
    python src/models/predict_model.py
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import pickle
import sys
from pathlib import Path


# Import params from params.yaml
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PARAMS, MONGO_URI


# ==================== Step 1 ====================
# Load trained model

model_path = Path('models/xgboost_model.pkl')

if not model_path.exists():
    print(f'ERROR: Model not found at {model_path}')
    print('Please train the model first: python src/models/train_model.py')
    sys.exit(1)

with open(model_path, 'rb') as f:
    model = pickle.load(f)

print(f'Step 1: Model loaded from {model_path}')


# ==================== Step 2 ====================
# Load test data 

X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv')

print('Step 2: Test data loaded.')
print(f'X_test: {X_test.shape}')
print(f'y_test: {y_test.shape}')


# ==================== Step 3 ====================
# Make predictions

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print(f'Step 3: Predictions made for {len(y_pred)} samples.')


# ==================== Step 4 ====================
# Evaluate predictions

f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)


print(f'\nStep 4: Model Performance:')
print(f'  F1-Score:  {f1:.4f}')
print(f'  Recall:    {recall:.4f}')
print(f'  Precision: {precision:.4f}')
print(f'  Accuracy:  {accuracy:.4f}')
print(f'  ROC-AUC:   {roc_auc:.4f}')

print('\nConfusion Matrix:')
print(confusion_matrix(y_test, y_pred))

print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=['No Rain', 'Rain']))
