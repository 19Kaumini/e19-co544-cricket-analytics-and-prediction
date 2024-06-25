import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.base import ClassifierMixin
from zenml import step, pipeline
from zenml.client import Client

# Set up MLflow tracking
import dagshub
dagshub.init(repo_owner='chandula00', repo_name='e19-co544-cricket-analytics-and-prediction', mlflow=True)
mlflow.set_experiment("Match NRR Prediction")

def get_cross_val_score(X: np.ndarray, y: np.ndarray, model: ClassifierMixin, cv: int = 5):
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    mean_scores = np.mean(scores)
    std_devs = np.std(scores)
    return scores, mean_scores, std_devs

def get_clf_metrics(y_train: np.ndarray, y_pred_train: np.ndarray, y_test: np.ndarray, y_pred_test: np.ndarray):
    # Calculate mean squared error and R-squared
    mse_train = mean_squared_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    return mse_train, r2_train, mse_test, r2_test

@step
def load_data() -> pd.DataFrame:
    """Load a dataset."""
    data = pd.read_csv("Data/selected_data/all_batters_NRR.csv")
    return data

@step
def data_preparation(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target = 'net_run_rate'
    dataframe = data.copy()
    dataframe = pd.get_dummies(dataframe, columns=['batter', 'batting_team', 'bowling_team', 'venue',], dtype=int)
    y = dataframe[target]
    X = dataframe.drop(columns=[target,"won"])
    X_train, X_test, y_train, y_true = train_test_split(X, y, test_size=0.2, random_state=42)
    return X, y, X_train, X_test, y_train, y_true

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
def NRR_Predict_pipeline():
    data = load_data()        
    X_train, X_test, y_train, y_test = data_preparation(data)        
    model = train_rf(X_train, y_train)    
    recall_metric = evaluate_model(model, X_test, y_test)

    print(f"Recall: {recall_metric}")

experiment_tracker = Client().active_stack.experiment_tracker

def main():
    # Start a single MLflow run here
    with mlflow.start_run():
        NRR_Predict_pipeline()

if __name__ == '__main__':
    main()
