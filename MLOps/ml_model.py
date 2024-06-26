import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from zenml import step, pipeline
from zenml.client import Client
import joblib

# Set up MLflow tracking
# import dagshub
# dagshub.init(repo_owner='chandula00', repo_name='e19-co544-cricket-analytics-and-prediction', mlflow=True)
mlflow.set_experiment("Match NRR Prediction")

def get_cross_val_score(X: np.ndarray, y: np.ndarray, model: RandomForestRegressor, cv: int = 5):
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    mean_scores = np.mean(scores)
    std_devs = np.std(scores)
    return scores, mean_scores, std_devs

def get_clf_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    # Calculate mean squared error and R-squared
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return mse, r2

@step
def load_data() -> pd.DataFrame:
    """Load a dataset."""
    data = pd.read_csv("Data/selected_data/all_batters_NRR.csv")
    return data

@step
def data_preparation(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    target = 'net_run_rate'
    dataframe = data.copy()
    dataframe = pd.get_dummies(dataframe, columns=['batter', 'batting_team', 'bowling_team', 'venue'], dtype=int)
    y = dataframe[target]
    X = dataframe.drop(columns=[target, "won"])
    return X.to_numpy(), y.to_numpy()

@step(enable_cache=False)
def train_rf(X: np.ndarray, y: np.ndarray, model_name: str = "RandomForestRegressor", estimators: int = 100, RANDOM_STATE: int = 42) -> RandomForestRegressor:
    """Training a sklearn RF model."""
    model = RandomForestRegressor(n_estimators=estimators, random_state=RANDOM_STATE)
    model.fit(X, y)

    # Save the model with joblib
    joblib.dump(model, f"models/{model_name}.joblib")

    print(f"Model: {model_name} trained successfully.")
    print(f"Model Parameters: {model.get_params()}")
    
    # MLflow logging
    mlflow.log_param('n_estimators', estimators)
    mlflow.log_param('random_state', RANDOM_STATE)
    mlflow.sklearn.log_model(model, model_name)
    
    return model

@step(enable_cache=False)
def evaluate_model(model: RandomForestRegressor, X: np.ndarray, y: np.ndarray):
    """Model Evaluation and ML metrics register."""
    y_pred = model.predict(X)
    
    # metrics
    scores, mean_scores, std_devs = get_cross_val_score(X, y, model)
    mse, r2 = get_clf_metrics(y, y_pred)
    
    mlflow.log_metric("Mean_Squared_Error", mse)
    mlflow.log_metric("R2_Score", r2)
    mlflow.log_metric("cross_val_mean", mean_scores)
    mlflow.log_metric("cross_val_std_dev", std_devs)

    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")
    print(f"Cross Validation Scores: {scores}")
    print(f"Cross Validation Mean: {mean_scores}")
    print(f"Cross Validation Std Dev: {std_devs}")

@pipeline(enable_cache=False)
def NRR_Predict_pipeline():
    data = load_data()
    X, y = data_preparation(data)
    model = train_rf(X, y)
    evaluate_model(model, X, y)

experiment_tracker = Client().active_stack.experiment_tracker

def main():
    # Start a single MLflow run here
    with mlflow.start_run():
        NRR_Predict_pipeline()

if __name__ == '__main__':
    main()
