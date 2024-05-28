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
TEST_DATA_PATH = '2nd_inning_data.csv'

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

if __name__ == '__main__':
    pytest.main()
