import os
from flask_cors import CORS, cross_origin
import pandas as pd
import numpy as np
from joblib import load
from flask import Flask, request, jsonify, render_template

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

# Load CSV files
venues_df = pd.read_csv("train_venues.csv")
player_vs_bowler_df = pd.read_csv("train_player_vs_bowler.csv")
train_set = pd.read_csv("train_set.csv")
all_columns = pd.read_csv("Models/MultiOutputRegression/all_columns.csv")

class ClientApp:
    def __init__(self):
        self.batter_runs = self.load_batter_runs()
        self.strike_rate = self.load_strike_rate()
        self.final_team_score = self.load_final_team_score()
        self.net_run_rate = self.load_net_run_rate()


    def load_batter_runs(self):
        # paths for models Batter Runs, Strike Rate, Final Team Score, and Net Run Rate
        model_path_batter_runs = 'Models/MultiOutputRegression/pkls/batter_runs_model.pkl'
        return load(model_path_batter_runs)
    
    def load_strike_rate(self):
        model_path_strike_rate = 'Models/MultiOutputRegression/pkls/strike_rate_model.pkl'
        return load(model_path_strike_rate)
    
    def load_final_team_score(self):
        model_path_final_team_score = 'Models/MultiOutputRegression/pkls/final_team_total_model.pkl'
        return load(model_path_final_team_score)
    
    def load_net_run_rate(self):
        model_path_net_run_rate = 'Models/MultiOutputRegression/pkls/nrr_model.pkl'
        return load(model_path_net_run_rate)   

    def predict(self, input_data):
        predicted_batter_runs = self.batter_runs.predict(input_data)
        predicted_strike_rate = self.strike_rate.predict(input_data)
        predicted_final_team_score = self.final_team_score.predict(input_data)
        predicted_net_run_rate = self.net_run_rate.predict(input_data)
        return predicted_batter_runs, predicted_strike_rate, predicted_final_team_score, predicted_net_run_rate

def derive_features(input_dic: dict) -> dict:
	bowler_types = ['Left arm Fast', 'Right arm Fast', 'Left arm Orthodox',
					'Left arm Wrist spin', 'Right arm Legbreak', 'Right arm Offbreak']


	def get_venue_info():
		venue = input_dic['venue']
		del input_dic['venue']
		if venue in venues_df['venue'].values:  # Check if venue exists in DataFrame
			venue_mean_total = venues_df.loc[venues_df['venue'] == venue, 'total_mean']
			venue_first_bat_won_ratio = venues_df.loc[venues_df['venue']
													== venue, 'first_bat_won_ratio']
			# Access single value from Series
			input_dic['venue_mean_total'] = venue_mean_total.values[0]
			input_dic['venue_first_bat_won_ratio'] = venue_first_bat_won_ratio.values[0]
		else:
			input_dic['venue_mean_total'] = venues_df['total_mean'].median()
			input_dic['venue_first_bat_won_ratio'] = venues_df['first_bat_won_ratio'].median()


	def get_runs_remain():
		if input_dic['innings'] == 2:
			return
		if input_dic['innings'] == 1:
			return input_dic['venue_mean_total'] - input_dic['current_team_total']


	bowler_types = ['Left arm Fast', 'Right arm Fast', 'Left arm Orthodox',
					'Left arm Wrist spin', 'Right arm Legbreak', 'Right arm Offbreak']


	def get_bowler_type_data():
		player_vs_bowler = player_vs_bowler_df[player_vs_bowler_df['batter']
											== input_dic['batter']]
		player_vs_bowler.set_index("bowler_type", inplace=True)
		for bowler_type in bowler_types:
			if bowler_type in player_vs_bowler.index:
				input_dic[f'{bowler_type} Expected Runs'] = player_vs_bowler.loc[bowler_type,
																				'strike_rate'] / 100 * input_dic[bowler_type]
				input_dic[f'{bowler_type} Expected Wickets'] = input_dic[bowler_type] / \
					player_vs_bowler.loc[bowler_type, 'deliveries_per_wicket']
				input_dic[f'{bowler_type} Strike Rate'] = player_vs_bowler.loc[bowler_type, 'strike_rate']
				input_dic[f'{bowler_type} Deliveries Per Wicket'] = player_vs_bowler.loc[bowler_type,
																						'deliveries_per_wicket']

			else:
				input_dic[bowler_type] = 0
				input_dic[f'{bowler_type} Expected Runs'] = 0
				input_dic[f'{bowler_type} Expected Wickets'] = 0
				input_dic[f'{bowler_type} Strike Rate'] = 0
				input_dic[f'{bowler_type} Deliveries Per Wicket'] = 0




	def get_expected():
		bowler_types_expected_wickets_cols = [bowler_type + " Expected Wickets" for bowler_type in bowler_types]
		bowler_types_expected_runs_cols = [bowler_type + " Expected Runs" for bowler_type in bowler_types]

		input_dic['expected_wickets'] = sum([input_dic[key] for key in bowler_types_expected_wickets_cols])
		input_dic['expected_runs'] = sum([input_dic[key] for key in bowler_types_expected_runs_cols])

		input_dic['expected_runs'] = input_dic['expected_runs'] / np.sqrt(input_dic['expected_wickets']) 


	get_venue_info()
	get_runs_remain()
	get_bowler_type_data()
	get_expected()

	return input_dic

@app.route('/')
@cross_origin()
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    data = request.get_json()
    form_data = {
        'innings': int(data['innings']),
        'wickets_fallen': int(data['wickets_fallen']),
        'bowling_team': data['bowling_team'],
        'batting_team': data['batting_team'],
        'toss_winner': int(data['toss_winner']),
        'runs_remain': float(data['runs_remain']),
        'first_ball': int(data['first_ball']),
        'current_team_total': int(data['current_team_total']),
        'is_powerplay': data['is_powerplay'] == 'true',
        'Left arm Fast': float(data.get('Left arm Fast', 0)),
        'Left arm Orthodox': float(data.get('Left arm Orthodox', 0)),
        'Left arm Wrist spin': float(data.get('Left arm Wrist spin', 0)),
        'Right arm Fast': float(data.get('Right arm Fast', 0)),
        'Right arm Legbreak': float(data.get('Right arm Legbreak', 0)),
        'Right arm Offbreak': float(data.get('Right arm Offbreak', 0)),
        'venue': data['venue']
    }

    # print(form_data)

    input_dic = form_data.copy()

    batters_list = data['batter'].split(',')

    data = pd.DataFrame()

    for batter in batters_list:
        batter_dic = input_dic.copy()
        batter_dic['batter'] = batter
        new_features = derive_features(batter_dic)
    
        print(new_features)
        data = pd.concat([data,pd.DataFrame([new_features])])

    data.replace(np.inf,120,inplace=True)
    data.reset_index(inplace=True, drop=True)
    data_processed = pd.get_dummies(data=data,dtype=int)


    all_cols = all_columns.columns.tolist()
    data_cols = data_processed.columns.tolist()


    missing_columns = set(all_cols) - set(data_cols)

    for col in missing_columns:
        data_processed[col] = 0

        
    # Make predictions
    nrr_pred = nrr_model.predict(data_processed)
    sr_pred = sr_model.predict(data_processed)
    batter_run_pred = batter_run_model.predict(data_processed)
    final_team_total_pred = final_team_total_model.predict(data_processed)


    # Add predictions as new columns
    data["nrr_predicted"] = nrr_pred
    data["sr_predicted"] = sr_pred
    data["batter_run_predicted"] = batter_run_pred
    data["final_team_total_predicted"] = final_team_total_pred

    # Now data_processed contains the original features and the predictions


    data.sort_values(by='nrr_predicted', ascending=False, inplace=True)
    data.reset_index(inplace=True, drop=True)
    data.head()

    output = data.loc[0, ['batter', 'nrr_predicted', 'sr_predicted', 'batter_run_predicted', 'final_team_total_predicted']]
    return output.to_json()
    
    

    
    # # print the dattypes in data
    # # print(data_processed.dtypes)
    # predictions = client_app.predict(data_processed)
    # # concat the predictions to the data
    # data['batter_runs'] = predictions[0]
    # data['strike_rate'] = predictions[1]
    # data['final_team_score'] = predictions[2]
    # data['net_run_rate'] = predictions[3]

    # results = data[['batter','batter_runs','strike_rate','final_team_score','net_run_rate']]

    # # sort the results by net run rate
    # results = results.sort_values(by='net_run_rate',ascending=False)

nrr_model = load("Models/MultiOutputRegression/pkls/nrr_model.pkl")
batter_run_model = load("Models/MultiOutputRegression/pkls/batter_runs_model.pkl")
final_team_total_model = load("Models/MultiOutputRegression/pkls/final_team_total_model.pkl")
sr_model = load("Models/MultiOutputRegression/pkls/strike_rate_model.pkl")

if __name__ == '__main__':
    app.run(debug=True)
