"""
Train XGBoost model for rain prediction in Australia.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import pickle
import sys
from pathlib import Path
import argparse
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
import dagshub

# Import params
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PARAMS

# ==================== MLflow config ====================
EXPERIMENT_NAME = "WeatherAUS_YearBased_Training"
MODEL_NAME = "RainTomorrow_XGBoost"

dagshub.init(
    repo_owner="jooooo3141",
    repo_name="dec25bmlops_int_weather",
    mlflow=True
)
mlflow.set_experiment(EXPERIMENT_NAME)


# ==================== Data loading ====================
def load_split_data(split_id: int):
    splits_dir = Path("data/training_data_splits_by_year")
    split_dirs = list(splits_dir.glob(f"split_{split_id:02d}_*"))

    if not split_dirs:
        raise FileNotFoundError(f"Split {split_id} not found!")

    split_dir = split_dirs[0]
    split_name = split_dir.name

    X_train = pd.read_csv(split_dir / "X_train.csv")
    y_train = pd.read_csv(split_dir / "y_train.csv")
    X_test = pd.read_csv(split_dir / "X_test.csv")
    y_test = pd.read_csv(split_dir / "y_test.csv")

    years = split_name.split("_")[-1]

    split_info = {
        "split_id": split_id,
        "split_name": split_name,
        "years": years,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }

    print(f"Loaded {split_name}")
    return X_train, X_test, y_train, y_test, split_info


# ==================== NEW: Best model logic ====================
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

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            "tags.best_model = 'true' AND "
            #f"tags.dataset_version = '{dataset_version}' AND "
            "tags.model_type = 'xgboost'"
        ),
        max_results=1,
    )

    best_run = runs[0] if runs else None

    new_run = client.get_run(run_id)
    new_metric = new_run.data.metrics.get(metric_name)

    if new_metric is None:
        print(f"Metric {metric_name} missing – skipping tagging.")
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

        if best_run:
            client.delete_tag(best_run.info.run_id, "best_model")
            client.set_tag(best_run.info.run_id, "stage", "previous_champion")

        client.set_tag(run_id, "best_model", "true")
        client.set_tag(run_id, "stage", "champion")
    else:
        client.set_tag(run_id, "stage", "candidate")


# ==================== Training ====================
def train_model(split_id: int):
    X_train, X_test, y_train, y_test, split_info = load_split_data(split_id)

    run_name = f"split_{split_id:02d}_{split_info['years']}"
    mlflow_run = mlflow.start_run(run_name=run_name)

    try:
        # -------- Tags required for best-model logic --------
        mlflow.set_tag("dataset_version", split_info["years"])
        mlflow.set_tag("model_type", "xgboost")

        # -------- Params --------
        mlflow.log_param("split_id", split_id)
        mlflow.log_param("years", split_info["years"])
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))

        # -------- SMOTE --------
        smote = SMOTE(random_state=PARAMS["data"]["random_state"])
        X_train_smote, y_train_smote = smote.fit_resample(
            X_train, y_train.values.ravel()
        )

        mlflow.log_param("smote_applied", True)
        mlflow.log_param("train_samples_after_smote", len(X_train_smote))

        # -------- Model --------
        model_params = PARAMS["model"]
        mlflow.log_params(model_params)

        model = xgb.XGBClassifier(
            **model_params,
            eval_metric="logloss",
            use_label_encoder=False
        )
        model.fit(X_train_smote, y_train_smote)

        # -------- Evaluation --------
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "f1_score": f1_score(y_test, y_pred),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        mlflow.log_metrics(metrics)

        mlflow.log_text(
            str(confusion_matrix(y_test, y_pred)),
            "confusion_matrix.txt"
        )
        mlflow.log_text(
            classification_report(y_test, y_pred),
            "classification_report.txt"
        )

        # -------- Model logging --------
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            signature=mlflow.models.infer_signature(X_train, y_pred),
            input_example=X_train.iloc[:5],
        )

        # -------- NEW: best model comparison --------
        update_best_model_tag(
            run_id=mlflow_run.info.run_id,
            metric_name="f1_score",
            dataset_version=split_info["years"],
            experiment_name=EXPERIMENT_NAME,
        )

        # -------- Save pickle --------
        Path("models").mkdir(exist_ok=True)
        with open("models/xgboost_model.pkl", "wb") as f:
            pickle.dump(model, f)

        return metrics["f1_score"]

    finally:
        mlflow.end_run()


# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_id", type=int, required=True)
    args = parser.parse_args()

    f1 = train_model(args.split_id)
    print(f"Training finished – F1: {f1:.4f}")


if __name__ == "__main__":
    main()