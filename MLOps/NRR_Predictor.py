#!/usr/bin/env python
# coding: utf-8

# In[18]:

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import pandas as pd

# Load the data
data = pd.read_csv('../Data/selected_data/processed_data_NRR.csv')
data.info()


# Selecting Player "KM Mendis"

# In[19]:


# sl_batsmen = data[data['batting_team'] == "Sri Lanka"]['batter'].unique()
# sl_batsmen


# In[20]:


all_batsmen = data['batter'].unique()
all_batsmen


# In[21]:


# SL ICC T20 Team
# selected_batters = ["PWH de Silva",'KIC Asalanka','BKG Mendis',"P Nissanka",'PHKD Mendis','S Samarawickrama','AD Mathews','MD Shanaka','DM de Silva','M Theekshana','PVD Chameera','N Thushara','M Pathirana','D Madushanka']


# In[22]:


def getPlayerScores(player_name: str, innings: list[int] = [1, 2] ) -> pd.DataFrame:
    # Get the data for BKG Mendis if batter is BKG Mendis or non-striker is BKG Mendis
	player_data = data.loc[
		((data['batter'] == player_name) | (data['non_striker'] == player_name)) & (data['innings'].isin(innings))
	]

	player_data.head()

	# 3 matches missing from the data
	# group data by match_id
	gp = player_data.groupby('match_id')
	cols = ['batting_team', 'batter', 'non_striker', 'batter_runs', 'balls_faced', 'wicket_type', 'won', 'innings', 'over', 'delivery', 'wickets_fallen','bowling_team','venue','net_run_rate']
	player_scores= gp.last().loc[:, cols]

	# get the first ball he faced or at non-striker
	first_ball = gp.first().loc[:, ['over', 'delivery', 'wickets_fallen']]
	first_ball['first_ball'] = (first_ball['over'] * 6 + first_ball['delivery']).astype(int)

	player_scores['first_ball'] = first_ball['first_ball']
	player_scores['wickets_fallen'] = first_ball['wickets_fallen']

	# when BKG Mendis is the non-striker when the last ball was bowled
	# The batter_runs and balls_faced are not his, but the on_strike batter's
	# So, we need to get the last ball he faced
	# he might not even have faced a ball

	# get the last ball he faced

	matches_non_striker = player_scores[player_scores['non_striker'] == player_name].index
	
	# Sometimes the player might not even have faced a single ball
	# Eg: Afghanistan_Sri Lanka_2022-11-01 MD Shanaka not out on the non strikers end

	player_scores.loc[matches_non_striker, ['batter_runs', 'balls_faced']] = [0, 0]
	
	
	# get the last batter == player_name row from gp data
	gp = player_data[(player_data['batter'] == player_name) & (player_data['match_id'].isin(matches_non_striker))].groupby(['match_id'])
	last_batter_scores = gp.last()[['batter_runs', 'balls_faced']]	
	
	# update the rows with non_striker with correct values
	player_scores.update(last_batter_scores)
	
	



	# adding new features
	# strike rate
	player_scores['strike_rate'] = round(player_scores['batter_runs'] / player_scores['balls_faced'] * 100, 2)
	player_scores['out'] = player_scores['wicket_type'] != '0'
	player_scores['last_ball'] = (player_scores['over'] * 6 + player_scores['delivery']).astype(int)
 
 
	player_scores['batter'] = player_name
	player_scores.drop('non_striker', inplace=True, axis = 1)

	# drop over and delivery
	player_scores.drop(['over', 'delivery'], inplace=True, axis=1)
 
	return player_scores


# In[23]:


# getPlayerScores('BKG Mendis')

# merged_df = pd.DataFrame()

# for player in selected_batters:
#     player_scores = getPlayerScores(player)
    
#     merged_df = pd.concat([merged_df, player_scores])



# In[24]:


merged_df = pd.DataFrame()

for player in all_batsmen:
    player_scores = getPlayerScores(player)
    
    merged_df = pd.concat([merged_df, player_scores])


# In[25]:


merged_df.info()


# In[26]:


merged_df.loc[merged_df["strike_rate"].isnull()]


# In[27]:


merged_df["strike_rate"] = merged_df["strike_rate"].fillna(0)


# In[28]:


merged_df.info()


# In[29]:


merged_df.head()


# In[30]:


merged_df.info()


# In[31]:


merged_df.drop(columns=["batter_runs","balls_faced","wicket_type","strike_rate","out","last_ball"],inplace=True)


# In[32]:


merged_df.info()


# In[33]:


merged_df.to_csv('./all_batters_NRR.csv',index=False)


# In[35]:


from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
RANDOM_STATE = 42
data = merged_df

target = 'net_run_rate'

# Preprocess the data
data = pd.get_dummies(data, columns=['batter', 'batting_team', 'bowling_team', 'venue',], dtype=int)
data.head()

y = data[target]
X = data.drop(columns=[target,"won"])

model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

# Print the mean and standard deviation of the cross-validation scores
print('Cross-validation scores:', scores)
print('Mean cross-validation score:', scores.mean())
print('Standard deviation of cross-validation scores:', scores.std())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)


# Calculate mean squared error and R-squared
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print('Train - Mean Squared Error:', mse_train)
print('Train - R-squared:', r2_train)
print('Test - Mean Squared Error:', mse_test)
print('Test - R-squared:', r2_test)


# In[ ]:


X_test.head()


# In[ ]:


win_proba_df = pd.DataFrame(win_probability, columns=['win_probability'])

X_test = X_test.reset_index(drop=True)
win_proba_df = win_proba_df.reset_index(drop=True)
result = pd.concat([X_test, win_proba_df], axis=1)

result.head()


# In[ ]:


y_train.value_counts()


# In[ ]:


y_test.value_counts()

# Save the model
joblib.dump(model, 'models/model.pkl')