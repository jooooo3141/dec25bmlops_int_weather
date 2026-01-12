"""
Preprocessing of raw weather data.

This script takes the raw Kaggle dataset and performs initial cleaning and feature engineering.

Input:  data/raw/weatherAUS.csv
Output: data/interim/df_preprocessed.csv, data/processed/X_train.csv, data/processed/y_train.csv, data_processed/X_test.csv, data/processed/y_test.csv

Usage:
    python src/data/preprocess
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

# Import params from config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PARAMS

# Use parameters from params.yaml
TEST_SIZE = PARAMS['data']['test_size']
RANDOM_STATE = PARAMS['data']['random_state']

# Import raw data
df = pd.read_csv('data/raw/weatherAUS.csv')

print('Data imported: data/raw/weatherAUS.csv')
print('Columns present: ', df.columns)


# ==================== Preprocessing Step 1 ====================
# Delete rows containing  missing values in RainTomorrow (target variable), Rainfall and RainToday columns

df = df.dropna(subset=['RainTomorrow', 'Rainfall', 'RainToday'])
print('Preprocessing Step 1: Deleted rows containing missing values in RainTomorrow, Rainfall and RainToday columns.')


# ==================== Preprocessing Step 2 ====================
# Delete columns: 'Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm' as they contain more than 30% of missing data

df = df.drop(['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm'], axis=1)
print('Preprocessing Step 2: Deleted columns: Evaporation, Sunshine, Cloud9am, Cloud3pm as they contain more than 30% of missing data.')


# ==================== Preprocessing Step 3 ====================
# Change Date column to datetime type and separate year, month, day in individual columns

df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# Function to extract the season in Australia from month
def get_season_aus(month):
    """
    Australian seasons:
    Summer: Dec, Jan, Feb (12, 1, 2)
    Autumn: Mar, Apr, May (3, 4, 5)
    Winter: Jun, Jul, Aug (6, 7, 8)
    Spring: Sep, Oct, Nov (9, 10, 11)
    """
    if month in [12, 1, 2]:
        return 'Summer'
    elif month in [3, 4, 5]:
        return 'Autumn'
    elif month in [6, 7, 8]:
        return 'Winter'
    else:  # 9, 10, 11
        return 'Spring'

# Generate Season column
df['Season'] = df['Month'].apply(get_season_aus)

# Drop Date and Month column as we have now the Season and Year column for modeling
df = df.drop(['Date', 'Month'], axis=1)

print('Preprocessing Step 3: Season colum added from Date column. Deleted Date column.')
print('Missing values: ', df.isnull().sum())


# ==================== Preprocessing Step 4 ====================
# Change in column RainTomorrow No to 0 and Yes to 1 -> numerical values
# Change in column RainToday No to 0 and Yes to 1 -> numerical values

df['RainTomorrow'] = df['RainTomorrow'].map({'No': 0, 'Yes': 1})
df['RainToday'] = df['RainToday'].map({'No': 0, 'Yes': 1})
print('Preproecssing Step 4: Changed in RainTomorrow and RainToday columns No to 0 and Yes to 1.')


# Save dataset in interim folder
df.to_csv('data/interim/df_preprocessed.csv', index=False)

print('Saved: data/interim/df_preprocessed.csv')
print('Columns: ', df.columns)
print('Missing values: ', df.isnull().sum())


# ========== Preprocessing Step 5 ==========
# Split into X and y before filling missing values to prevent data leakage

X = df.drop('RainTomorrow', axis=1)
y = df['RainTomorrow']


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print('Preprocessing Step 5: Generated train/test split (80%, 20%): X_train, X_test, y_train, y_test with X being columns and y being target.')


# ========== Preprocessing Step 6 ==========
# Fill missing values for numerical columns with median
# Numerical columns
numeric_cols = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed',
                'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
                'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']

# Compute median from training data and fill missing values of train and test set to prevent data leakage (information from test set leaking into training set)
train_medians = X_train[numeric_cols].median()

X_train[numeric_cols] = X_train[numeric_cols].fillna(train_medians)
X_test[numeric_cols] = X_test[numeric_cols].fillna(train_medians)


# Fill missing values for categorical columns with modus
# Categorical columns'
categorical_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm']

# Compute modus from trainng data and fill missing values of train and test set to prevent data leakage
train_modes = X_train[categorical_cols].mode().iloc[0]

X_train[categorical_cols] = X_train[categorical_cols].fillna(train_modes)
X_test[categorical_cols] = X_test[categorical_cols].fillna(train_modes)

print('Preprocessing Step 6:')
print('Missing values in numerical columns filled with medians of traning set.')
print('Missing values in categorical columns filled with modus of training set.')
print('Missing values after filling in training set: ', X_train.isnull().sum())
print('Missing values after filling in test set: ', X_test.isnull().sum())


# ==================== Preprocessing Step 7 ====================
# One-Hot Encoding to achieve only numerical values in dataset

# Categoerical columns
categorical_cols_encoding = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'Season']

# One-Hot Encoding training set
X_train = pd.get_dummies(X_train, columns=categorical_cols_encoding, drop_first=True)

# One-Hot Encoding for test set
X_test = pd.get_dummies(X_test, columns=categorical_cols_encoding, drop_first=True)

print('Preprocessing Step 7: One-Hot Encoding for categorical columns: Location, WindGustDir, WindDir9am, WindDir3pm, Season')


# ==================== Preprocessing Step 8 ====================
# Convert boolean to integer

# Find boolean columns
bool_cols = X_train.select_dtypes(include=['bool']).columns

# Convert to integer (True -> 1, False -> 0)
X_train[bool_cols] = X_train[bool_cols].astype(int)
X_test[bool_cols] = X_test[bool_cols].astype(int)

print('Preprocessing Step 8: Changed boolean to integers.')


# ==================== Preprocessing Step 9 ====================
# Scaling of numerical columns using StandardScaler

numerical_cols_scale = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed',
                            'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
                            'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']

# Generate scaler
scaler = StandardScaler()

scaler.fit(X_train[numerical_cols_scale])

X_train[numerical_cols_scale] = scaler.transform(X_train[numerical_cols_scale])
X_test[numerical_cols_scale] = scaler.transform(X_test[numerical_cols_scale])

print('Preprocessing Step 8: Scaling of numerical columns using StandardScaler.')


# Save modeling datasets in data/processed
X_train.to_csv('data/processed/X_train.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False, header=True)

X_test.to_csv('data/processed/X_test.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False, header=True)

print('Saved X_train, y_train, X_test, y_test in data/processed. Data ready for modeling.')


# Check preprocessed data

print('Check preprocessed data:')

# 1. Shape Check
print(f"Train Shape: {X_train.shape}")
print(f"Test Shape:  {X_test.shape}")
assert X_train.shape[1] == X_test.shape[1], "Column mismatch!"
print(f"Columns match: {X_train.shape[1]} features\n")

# 2. Missing Values
train_missing = X_train.isnull().sum().sum()
test_missing = X_test.isnull().sum().sum()
print(f"Missing Values - Train: {train_missing}, Test: {test_missing}\n")

# 3. Data Types
print("Data types:")
print(X_train.dtypes.value_counts())

# 4. Column Names Check
assert list(X_train.columns) == list(X_test.columns), "Column names don't match!"
print("Column names match\n")

# 5. Class distribution of target
print('Class distribution y_train: ', (y_train.value_counts()/y_train.value_counts().sum())*100)
print('Class distribution y_test: ', (y_test.value_counts()/y_test.value_counts().sum())*100)
print('Imbalanced class distribution.')
