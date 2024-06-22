import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle

# Load the data
data = pd.read_csv('./Data/selected_data/processed_data.csv')

# merging medium bowlers to fast
data.loc[data['bowler_type'] == 'Left arm Medium', 'bowler_type'] = 'Left arm Fast'
data.loc[data['bowler_type'] == 'Right arm Medium', 'bowler_type'] = 'Right arm Fast'
data['date'] = data['match_id'].apply(lambda x: x[-10: ])
data['date'] = pd.to_datetime(data['date'])
data.info()

data.iloc[:, 10:].head(20)

def getPlayerScores(player_name: str, innings: list[int] = [1, 2]) -> pd.DataFrame:
    # Get the data for BKG Mendis if batter is BKG Mendis or non-striker is BKG Mendis
	player_data = data.loc[
		((data['batter'] == player_name) | (data['non_striker'] == player_name)) & (
		    data['innings'].isin(innings))
	]

	player_data.head()

	# group data by match_id
	gp = player_data.groupby('match_id')
	cols = ['date', 'batter', 'non_striker', 'batter_runs', 'balls_faced', 'wicket_type', 'won', 'innings', 'over',
	    'delivery', 'wickets_fallen', 'bowling_team', 'batting_team', 'venue', 'toss_winner', 'batter_type', 'non_striker_type']
	player_scores = gp.last().loc[:, cols]

	# get the first ball he faced or at non-striker
	first_ball = gp.first().loc[:, ['over', 'delivery',
	                      'wickets_fallen', 'current_team_total']]
	first_ball['first_ball'] = (
	    first_ball['over'] * 6 + first_ball['delivery']).astype(int)

	player_scores['first_ball'] = first_ball['first_ball']
	player_scores['wickets_fallen'] = first_ball['wickets_fallen']
	player_scores['current_team_total'] = first_ball['current_team_total']
	player_scores['is_powerplay'] = first_ball['first_ball'] <= 30

    # convert toss win to boolean
	player_scores['toss_winner'] = (
	    player_scores['toss_winner'] == player_scores['batting_team']).astype(int);

    # when BKG Mendis is the non-striker when the last ball was bowled
	# The batter_runs and balls_faced are not his, but the on_strike batter's
	# So, we need to get the last ball he faced
	# he might not even have faced a ball

	# get the last ball he faced

	matches_non_striker = player_scores[player_scores['non_striker']
	    == player_name].index

	# Sometimes the player might not even have faced a single ball
	# Eg: Afghanistan_Sri Lanka_2022-11-01 MD Shanaka not out on the non strikers end

	player_scores.loc[matches_non_striker, [
	    'batter_runs', 'balls_faced']] = [0, 0]

	# get the last batter == player_name row from gp data
	gp = player_data[(player_data['batter'] == player_name) & (
	    player_data['match_id'].isin(matches_non_striker))].groupby(['match_id'])
	last_batter_scores = gp.last()[['batter_runs', 'balls_faced']]

	# update the rows with non_striker with correct values
	player_scores.update(last_batter_scores)

	# adding new features
	# strike rate
	player_scores['strike_rate'] = round(
	    player_scores['batter_runs'] / player_scores['balls_faced'] * 100, 2)
	player_scores['out'] = player_scores['wicket_type'] != '0'
	player_scores['last_ball'] = (
	    player_scores['over'] * 6 + player_scores['delivery']).astype(int)

	# drop over and delivery
	player_scores.drop(['over', 'delivery'], inplace=True, axis=1)

	# concatenating the remaining bowler types number to the dataset
	matches = data[data['match_id'].isin(player_scores.index)]
	# matches = matches[matches['batting_team'] == 'Sri Lanka']
	cols = ['match_id', 'batter', 'non_striker', 'bowler_type', 'batter_runs',
	    'balls_faced', 'wicket_type', 'won', 'innings', 'over', 'delivery', 'wickets_fallen']
	matches = matches[cols]
	matches['ball_number'] = (matches['over'] * 6 +
	                          matches['delivery']).astype(int)
	matches.drop(['over', 'delivery'], inplace=True, axis=1)

	def filter_by_player_and_ball_number(group):
		player_data = group[group['batter'] == player_name]

		if player_data.empty:
			return player_data.drop('match_id', axis=1)

		first_ball_number = player_data['ball_number'].iloc[0]

		# return player_data[player_data['ball_number'] > first_ball_number].drop('match_id', axis=1) # This is for number of balls faced

		# fixed error should be greater or equal
		remaining = group[group['ball_number'] >= first_ball_number].drop(
		    'match_id', axis=1)  # return the remianing number of deliveries for each type
		return remaining

	gp = matches.groupby('match_id').apply(filter_by_player_and_ball_number)
	remaining_ball_types = gp.groupby(
	    'match_id')['bowler_type'].value_counts().unstack(fill_value=0)
	remaining_ball_types = remaining_ball_types.reset_index()

	player_scores = player_scores.merge(
	    remaining_ball_types, how='left', on='match_id')

	player_scores.fillna(0, inplace=True)
	
    
    # if batsman ended as non_striker, make him the batter and drop the nonstriker column
	player_scores['batter'] = player_name
	player_scores.drop('non_striker', inplace=True, axis = 1)



	# Sort according to date
	player_scores.sort_values(by='date', inplace=True)

	def calculate_recent_form(row, window=100):
		"""
		Calculates the average runs scored by the player in the last 'window' days (excluding the current date).
		"""
		date = row['date']
		df = player_scores.copy()
		df = df[df['date'] < date]
		df = df[df['date'] >= (date - pd.Timedelta(days=window))]
		average_runs = df['batter_runs'].mean() if len(df) > 0 else 0
		return average_runs	

	player_scores['recent_form'] = player_scores.apply(calculate_recent_form, axis=1)
	
	# reindex
	player_scores.reset_index(drop=True, inplace=True)

	return player_scores




# selected_batters = ["PWH de Silva",'KIC Asalanka','BKG Mendis',"P Nissanka",'PHKD Mendis','S Samarawickrama','AD Mathews','MD Shanaka','DM de Silva','M Theekshana','PVD Chameera','N Thushara','M Pathirana','D Madushanka']

merged_df = pd.DataFrame()

for player in data['batter'].unique():
    print("Analyzing Player", player)
    
    player_scores = getPlayerScores(player)
    merged_df = pd.concat([merged_df, player_scores])

merged_df.to_csv('merged_df_checkpoint.csv')

import pandas as pd
import numpy as np

merged_df = pd.read_csv('merged_df_checkpoint.csv')
merged_df.reset_index(inplace=True)
merged_df.info()

merged_df.drop(['index','Unnamed: 0'], axis = 1, inplace = True)
merged_df.head()

# Drop all the records of lower order batsman (6 down and onwards)
merged_df = merged_df[merged_df['wickets_fallen'] < 6]

# Drop all the records of opening batsmen
merged_df = merged_df[merged_df['wickets_fallen'] != 0]

merged_df.reset_index(drop=True, inplace=True)

merged_df['batter_runs_cat'] = pd.cut(merged_df['batter_runs'], bins = [-1, 4, 15, 30, np.inf], labels = ['0-4', '5-15', '15-30', '30+'])
merged_df['batter_runs_cat'].hist()

from sklearn.model_selection import StratifiedShuffleSplit

# Split Train Test Data Sets
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state = 42);
for train_index, test_index in split.split(merged_df, merged_df['batter_runs_cat']):
    strat_train_set = merged_df.loc[train_index]
    strat_test_set = merged_df.loc[test_index]

print(strat_test_set['batter_runs_cat'].value_counts() / len(strat_test_set));

# Drop the categorical column used for strat
for set_ in (strat_test_set, strat_train_set):
    set_.drop('batter_runs_cat', axis = 1, inplace = True)


strat_train_set.to_csv("strat_train_set.csv")
strat_test_set.to_csv("strat_test_set.csv")

def get_player_v_bowlers(player_name: str, match_ids, innings=[1, 2], ) -> pd.DataFrame:
    player_data = data.loc[
        (data['batter'] == player_name) & (data['innings'].isin(
            innings)) & (data['match_id'].isin(match_ids))
    ]

    player_data.head()

    # Convert all medium bowlers to fast
    player_data.loc[player_data['bowler_type'] ==
                    'Left arm Medium', 'bowler_type'] = 'Left arm Fast'
    player_data.loc[player_data['bowler_type'] ==
                    'Right arm Medium', 'bowler_type'] = 'Right arm Fast'

    player_data['out'] = (player_data['wicket_type'] != '0') & (
        player_data['wicket_type'] != 'run out')

    cols = ['batter', 'non_striker', 'runs_by_bat', 'out',
            'won', 'innings', 'over', 'delivery', 'bowler_type']
    player_data = player_data[cols]

    gp = player_data.groupby('bowler_type')

    player_v_bowler = pd.DataFrame()
    player_v_bowler['strike_rate'] = round(gp['runs_by_bat'].mean() * 100, 3)
    player_v_bowler['strike_rate_std'] = gp['runs_by_bat'].std()
    player_v_bowler['wickets'] = gp['out'].sum()
    player_v_bowler['deliveries_per_wicket'] = round(1 / gp['out'].mean(), 3)
    player_v_bowler['deliveries'] = gp.size()
    return player_v_bowler


# for player in data['batter'].unique():

#     # Attaching the player_v_bowler results to this dataset
#     print(f"Updating {player}...")
#     print(strat_train_set[strat_train_set['batter' == player]]);
#     player_scores = strat_train_set[strat_train_set['batter' == player]]

#     player_vs_bowler = get_player_v_bowlers(player_name, player_scores.index)

#     bowler_types = ['Left arm Fast', 'Right arm Fast', 'Left arm Orthodox',
#                     'Left arm Wrist spin', 'Right arm Legbreak', 'Right arm Offbreak']
#     for bowler_type in bowler_types:
#         if bowler_type in player_vs_bowler.index:
#             player_scores[f'{bowler_type} Expected Runs'] = player_vs_bowler.loc[bowler_type,
#                                                                                  'strike_rate'] / 100 * player_scores[bowler_type]
#             player_scores[f'{bowler_type} Expected Wickets'] = player_scores[bowler_type] / \
#                 player_vs_bowler.loc[bowler_type, 'deliveries_per_wicket']
#             player_scores[f'{bowler_type} Strike Rate'] = player_vs_bowler.loc[bowler_type, 'strike_rate']
#             player_scores[f'{bowler_type} Deliveries Per Wicket'] = player_vs_bowler.loc[bowler_type,
#                                                                                      'deliveries_per_wicket']
#         else:
#             player_scores[bowler_type] = 0
#             player_scores[f'{bowler_type} Expected Runs'] = 0
#             player_scores[f'{bowler_type} Expected Wickets'] = 0
#             player_scores[f'{bowler_type} Strike Rate'] = 0
#             player_scores[f'{bowler_type} Deliveries Per Wicket'] = 0

#     # strat_train_set.update(player_scores)

match_ids_in_train_set = strat_train_set['match_id']
def attach_bowler_type_stats(player_scores):
    
    player_name = player_scores['batter'].iloc[0]
    player_vs_bowler = get_player_v_bowlers(player_name, match_ids = match_ids_in_train_set);
    
    bowler_types = ['Left arm Fast', 'Right arm Fast', 'Left arm Orthodox',
                    'Left arm Wrist spin', 'Right arm Legbreak', 'Right arm Offbreak']
    for bowler_type in bowler_types:
        if bowler_type in player_vs_bowler.index:
            player_scores[f'{bowler_type} Expected Runs'] = player_vs_bowler.loc[bowler_type,
                                                                                 'strike_rate'] / 100 * player_scores[bowler_type]
            player_scores[f'{bowler_type} Expected Wickets'] = player_scores[bowler_type] / \
                player_vs_bowler.loc[bowler_type, 'deliveries_per_wicket']
            player_scores[f'{bowler_type} Strike Rate'] = player_vs_bowler.loc[bowler_type, 'strike_rate']
            player_scores[f'{bowler_type} Deliveries Per Wicket'] = player_vs_bowler.loc[bowler_type,
                                                                                     'deliveries_per_wicket']
        else:
            player_scores[bowler_type] = 0
            player_scores[f'{bowler_type} Expected Runs'] = 0
            player_scores[f'{bowler_type} Expected Wickets'] = 0
            player_scores[f'{bowler_type} Strike Rate'] = 0
            player_scores[f'{bowler_type} Deliveries Per Wicket'] = 0
            
    return player_scores

 # Appending bowler type data to the training set   
gp = strat_train_set.groupby('batter')
result = gp.apply(attach_bowler_type_stats);

strat_train_set = result.reset_index(drop=True)

# Appending the bowler type data to the testing set -> Only the training summaries are used (test set averages, strikerates are not used)
result = gp.apply(attach_bowler_type_stats);
strat_test_set = result.reset_index(drop=True)

# Saving checkpoint
strat_train_set.to_csv("strat_train_set.csv")
strat_test_set.to_csv("strat_test_set.csv")

# Additional Features Experimenting

# bowler_types = ['Left arm Fast', 'Right arm Fast', 'Left arm Orthodox', 'Left arm Wrist spin', 'Right arm Legbreak', 'Right arm Offbreak']
# bowler_types_expected_wickets_cols = [bowler_type + " Expected Wickets" for bowler_type in bowler_types]
# bowler_types_expected_runs_cols = [bowler_type + " Expected Runs" for bowler_type in bowler_types]

# strat_train_set['expected_wickets'] = strat_train_set.loc[:, bowler_types_expected_wickets_cols].sum(axis=1)
# strat_train_set['expected_runs'] = strat_train_set.loc[:, bowler_types_expected_runs_cols].sum(axis=1)
# strat_train_set['expected_runs'] = strat_train_set['expected_runs'] / np.sqrt(strat_train_set['expected_wickets']) 

# strat_test_set['expected_wickets'] = strat_test_set.loc[:, bowler_types_expected_wickets_cols].sum(axis=1)
# strat_test_set['expected_runs'] = strat_test_set.loc[:, bowler_types_expected_runs_cols].sum(axis=1)
# strat_test_set['expected_runs'] = strat_test_set['expected_runs'] / np.sqrt(strat_test_set['expected_wickets'])


# print(strat_train_set.loc[:, ['batter_runs', 'expected_runs']].corr())
# strat_train_set.loc[:, ['batter_runs', 'expected_runs']]

# Remove the columns that cannot be known by the ongoing match state
# Example: The out or wicket type is not known during the match
strat_test_set.drop(['wicket_type', 'out', 'match_id', 'won'], axis = 1, inplace = True)
strat_train_set.drop(['wicket_type', 'out', 'match_id', 'won'], axis = 1, inplace = True)


# fill 0 for NaN
strat_test_set.fillna(value=0, inplace=True)
strat_train_set.fillna(value=0, inplace=True)
# Saving checkpoint
strat_train_set.to_csv("strat_train_set.csv")
strat_test_set.to_csv("strat_test_set.csv")

import pandas as pd
import numpy as np

strat_train_set = pd.read_csv('strat_train_set.csv');

# Possible Targets: But we'll be only concerned with the number of runs scored for now
targets = ['batter_runs', 'balls_faced', 'last_ball', 'strike_rate']


# Match State -> Only data available in a given moment of a match. This will be the input
X = strat_train_set.drop(targets, axis=1)
y = strat_train_set['batter_runs']

# Replace infinity values by a high constant value
# These occured at deliveries per wicket column
# So infinity can be thought as the batsman never getting out
# i.e 120 balls faced, hence replacing inf with 120
X.replace([np.inf], 120, inplace=True);

# Preprocess the data
X_processed = pd.get_dummies(data=X,dtype=int)

X_processed.info()

X.drop('Unnamed: 0', inplace=True, axis=1)
X.columns

def display_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard Deviation: ", scores.std())

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

forest_reg = RandomForestRegressor(random_state=42)
scores = cross_val_score(forest_reg, X_processed, y, scoring="neg_mean_squared_error", cv = 10)


display_scores(np.sqrt(-scores))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor


# Split data into training and validating sets
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_val)

# Evaluate model performance
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred, alpha=0.7, label='Predicted vs. Actual')
plt.xlabel('Actual Target Values')
plt.ylabel('Predicted Target Values')
plt.title('Validation Set - Predicted vs. Actual Values')
plt.legend()
plt.grid(True)
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

model = XGBRegressor(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_processed_scaled = scaler.fit_transform(X_processed)
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Define the model architecture
model = keras.Sequential([
  layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),  # First hidden layer with 256 neurons and ReLU activation
  layers.Dense(256, activation='relu'),  # First hidden layer with 256 neurons and ReLU activation
  layers.Dense(128, activation='relu'),  # Second hidden layer with 128 neurons and ReLU activation
  layers.Dense(64, activation='relu'),  # Third hidden layer with 64 neurons and ReLU activation
  layers.Dense(32, activation='relu'),  # Third hidden layer with 64 neurons and ReLU activation
  layers.Dense(1)  # Output layer with 1 neuron for regression (single value prediction)
])

# Compile the model
model.compile(loss='mse', optimizer='adam')  # Mean squared error loss and Adam optimizer



# Train the model with validation split
history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_data=(X_val, y_val), verbose=1)  # Added validation data

# Access training and validation loss history for further analysis (optional)
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Save the model
model.save('models/trained_model')

# Optionally, save the model architecture to JSON and weights to HDF5
model_json = model.to_json()
with open("models/trained_model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("models/trained_model_weights.h5")

# Save the scaler using pickle
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


