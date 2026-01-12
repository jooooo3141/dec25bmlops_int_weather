"""
Train XGBoost model for rain prediction in Australia.
XGBoost was found as the best suited model for this binary classification task using LazyCLassifier on the cleaned dataset.
XGBoost parameters were identified using GridSearch.

This script loads processed data, applies SMOTE on training data for class balancing, trains an XGBoost classifier  and saves the trained model.

Input:  data/procesed/X_train, y_train
Output: models/xgboost_model.pkl

Usage:
    python src/models/train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import pickle
import sys
import mlflow
from mlflow.tracking import MlflowClient
import dagshub
import datetime
from pathlib import Path
import dagshub



# Import params from params.yaml
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PARAMS


# function for comparing trained model with existing best model in DagsHub MLflow registry
def update_best_model_tag(
    run_id: str,
    metric_name: str,
    dataset_version: str,
    experiment_name: str
):
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print("No experiment found – skipping best model tagging.")
        return

    # Suche aktuelles Bestmodell
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            f"tags.best_model = 'true' AND "
            f"tags.dataset_version = '{dataset_version}' AND "
            f"tags.model_type = 'xgboost'"
        ),
        max_results=1
    )

    best_run = runs[0] if runs else None

    new_run = client.get_run(run_id)
    new_metric = new_run.data.metrics.get(metric_name)

    if new_metric is None:
        print(f"Metric {metric_name} not found – skipping best model tagging.")
        return

    is_better = False

    if best_run is None:
        is_better = True
    else:
        best_metric = best_run.data.metrics.get(metric_name)
        if best_metric is None or new_metric > best_metric:
            is_better = True

    if is_better:
        print("New best model found!")

        # Altes Bestmodell enttaggen
        if best_run:
            client.delete_tag(best_run.info.run_id, "best_model")
            client.set_tag(best_run.info.run_id, "stage", "previous_champion")

        # Neues Bestmodell taggen
        client.set_tag(run_id, "best_model", "true")
        client.set_tag(run_id, "stage", "champion")
    else:
        client.set_tag(run_id, "stage", "candidate")

def load_split_data(split_id: int):
    splits_dir = Path("data/training_data_splits_by_year")
    split_dirs = list(splits_dir.glob(f"split_{split_id:02d}_*"))

    if not split_dirs:
        raise FileNotFoundError(
            f"Split {split_id} not found in {splits_dir}!\n"
            f"Run: python src/data/training_data_splits_by_year.py"
        )

    split_dir = split_dirs[0]
    split_name = split_dir.name

    X_train = pd.read_csv(split_dir / "X_train.csv")
    y_train = pd.read_csv(split_dir / "y_train.csv")
    X_test = pd.read_csv(split_dir / "X_test.csv")
    y_test = pd.read_csv(split_dir / "y_test.csv")

    year_info = split_name.split('_')[-1]
    
    print(f"Loaded Split {split_id}: {split_name}")
    print(f"Train: {len(X_train):6d} samples")
    print(f"Test:  {len(X_test):6d} samples")

    split_info = {
        'split_id': split_id,
        'split_name': split_name,
        'years': year_info,
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    return X_train, X_test, y_train, y_test, split_info

def train_model(split_id: int = None):
    # Get data version and experiment name (From params.yaml)?
    DATAVERSION = 'v1.0' #PARAMS['data']['dataversion']

    # set MLflow tracking URI
    #dagshub.init(repo_owner='a13x60r', repo_name='dec25bmlops_int_weather', mlflow=True)


    # ==================== Step 1 ====================
    # Load train data
    X_train, y_train, X_test, y_test, split_info = load_split_data(split_id)


    print('Step 1: Training data loaded')
    print(f'X_train: {X_train.shape}')
    print(f'y_train: {y_train.shape}')


    # ==================== Step 2 ====================
    # Apply SMOTE for class balancing on train data

    print('Step 2: SMOTE on training data to balance class distribution')
    print(f'Before SMOTE:')
    print(y_train.value_counts())

    smote = SMOTE(random_state=PARAMS['data']['random_state'])
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train.values.ravel())

    print(f'After SMOTE:')
    print(y_train_smote.value_counts())


    # ==================== Step 3 ====================
    # Train XGBoost model

    # Get model parameters from params.yaml
    model_params = {
        'max_depth': PARAMS['model']['max_depth'],
        'learning_rate': PARAMS['model']['learning_rate'],
        'n_estimators': PARAMS['model']['n_estimators'],
        'colsample_bytree': PARAMS['model']['colsample_bytree'],
        'subsample': PARAMS['model']['subsample'],
        'gamma': PARAMS['model']['gamma'],
        'min_child_weight': PARAMS['model']['min_child_weight'],
        'random_state': PARAMS['model']['random_state'],
        'eval_metric': 'logloss',
        'use_label_encoder': False
    }

    print('Step 3: Model training')
    print(f'Model parameters for XGBoost: {model_params}')

    dagshub.init(repo_owner='jooooo3141', repo_name='dec25bmlops_int_weather', mlflow=True)

    mlflow.set_experiment(PARAMS['mlflow']['experiment_name'])
    with mlflow.start_run(run_name='xgboost_rain_prediction_'+ datetime.datetime.now().strftime("%Y%m%d_%H%M%S")) as run:
        # Train model
        model = xgb.XGBClassifier(**model_params)
        model.fit(X_train_smote, y_train_smote,
                eval_set=[(X_test, y_test)],
                verbose=False)

        print('XGBoost model trained.')

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        mlflow.log_params(model_params)
        mlflow.log_metric('f1_score', f1)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('roc_auc', roc_auc)

        mlflow.set_tag("model_type", "xgboost")
        mlflow.set_tag("dataset_version", DATAVERSION)
        mlflow.xgboost.log_model(model, artifact_path="xgboost_model")

        update_best_model_tag(
            run_id=run.info.run_id,
            metric_name="roc_auc",
            dataset_version=DATAVERSION,
            experiment_name=PARAMS['mlflow']['experiment_name']
        )


    # ==================== Step 4 ====================
    # Save model

    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / 'xgboost_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f'Step 4: Model saved {model_path}')



train_model(split_id=1)