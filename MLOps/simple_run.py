import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin
from zenml import step, pipeline
from zenml.client import Client

# Set up MLflow tracking
import dagshub
dagshub.init(repo_owner='chandula00', repo_name='e19-co544-cricket-analytics-and-prediction', mlflow=True)
mlflow.set_experiment("Cardiovascular Disease Prediction")

def get_clf_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    return accuracy, precision, recall, f1

@step
def load_data() -> pd.DataFrame:
    """Load a dataset."""
    data = pd.read_csv("https://raw.githubusercontent.com/KattsonBastos/mlops-zenml-pipelines/main/data/cardio_train_sampled.csv")
    return data

@step
def data_preparation(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataframe = data.copy()
    dataframe['age'] = round(dataframe['age'] / 365).astype(int)
    X = dataframe.drop('cardio', axis=1).values
    y = dataframe['cardio'].values
    X_train, X_test, y_train, y_true = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_true

@step(enable_cache=False)
def train_rf(X_train: np.ndarray, y_train: np.ndarray, model_name: str = "model", max_depth: int = 4, random_state: int = 42) -> ClassifierMixin:
    """Training a sklearn RF model."""
    model = RandomForestClassifier(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    
    # MLflow logging
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('random_state', random_state)
    mlflow.sklearn.log_model(model, model_name)
    
    return model

@step(enable_cache=False)
def evaluate_model(model: ClassifierMixin, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Model Evaluation and ML metrics register."""
    y_pred = model.predict(X_test)
    
    # metrics
    accuracy, precision, recall, f1 = get_clf_metrics(y_test, y_pred)
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    
    return recall

@pipeline(enable_cache=False)
def training_rf_pipeline():
    data = load_data()        
    X_train, X_test, y_train, y_test = data_preparation(data)        
    model = train_rf(X_train, y_train)    
    recall_metric = evaluate_model(model, X_test, y_test)

    print(f"Recall: {recall_metric}")

experiment_tracker = Client().active_stack.experiment_tracker

def main():
    # Start a single MLflow run here
    with mlflow.start_run():
        training_rf_pipeline()

if __name__ == '__main__':
    main()
