from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the POST request
    data = request.json
    
    # Assuming data is in the same format as your training data
    df = pd.DataFrame(data)
    
    # Make prediction
    win_probability = model.predict_proba(df)[:, 1]  # Assuming binary classification for win probability
    
    # Return the result as JSON
    return jsonify({'win_probability': win_probability.tolist()})

# Define a route for a welcome message
@app.route('/')
def index():
    return 'Welcome to Cricket Analytics API!'

if __name__ == '__main__':
    app.run(debug=True)


