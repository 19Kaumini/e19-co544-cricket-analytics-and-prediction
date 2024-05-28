import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
import joblib

# Path to the model script
MODEL_PATH = '../Models/model.joblib'

# Sample data for testing
TEST_DATA_PATH = '../EDA/2nd_inning_data.csv'

@pytest.fixture
def sample_data():
    # Load the sample data
    df = pd.read_csv(TEST_DATA_PATH)
    return df

def test_data_loading(sample_data):
    # Test if data is loaded correctly
    assert not sample_data.empty, "The data should not be empty."
    assert 'winning_team' in sample_data.columns, "The column 'winning_team' should be present in the data."

def test_data_preprocessing(sample_data):
    # Test data preprocessing
    X = sample_data.drop(['winning_team', 'Index', 'venue'], axis=1)
    y = (sample_data['winning_team'] == sample_data['batting_team']).astype(int)
    
    assert X.shape[0] == y.shape[0], "The feature and target datasets should have the same number of samples."

def test_model_training(sample_data):
    # Test model training
    X = sample_data.drop(['winning_team', 'Index', 'venue'], axis=1)
    y = (sample_data['winning_team'] == sample_data['batting_team']).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=42))
    ])
    
    pipe.fit(X_train, y_train)
    y_pred_test = pipe.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred_test)
    assert accuracy > 0.5, "The model accuracy should be greater than 0.5."

def test_model_saving_loading(sample_data):
    # Test model saving and loading
    X = sample_data.drop(['winning_team', 'Index', 'venue'], axis=1)
    y = (sample_data['winning_team'] == sample_data['batting_team']).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=42))
    ])
    
    pipe.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(pipe, MODEL_PATH)
    
    # Load the model
    loaded_model = joblib.load(MODEL_PATH)
    y_pred_test = loaded_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred_test)
    assert accuracy > 0.5, "The loaded model accuracy should be greater than 0.5."

def test_model_predictions(sample_data):
    # Test model predictions
    X = sample_data.drop(['winning_team', 'Index', 'venue'], axis=1)
    y = (sample_data['winning_team'] == sample_data['batting_team']).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=42))
    ])
    
    pipe.fit(X_train, y_train)
    y_pred_proba = pipe.predict_proba(X_test)[:, 1]
    
    log_loss_value = log_loss(y_test, y_pred_proba)
    assert log_loss_value < 1.0, "The log loss should be less than 1.0."

if __name__ == '__main__':
    pytest.main()
