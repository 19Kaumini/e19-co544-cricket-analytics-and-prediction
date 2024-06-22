#!/usr/bin/env python
# coding: utf-8

# # Batting Order Model Development
# 
# This notebook will explore different models to get the win probability of a match given the state of the match and the next batsman to walk in to the crease. The input also contains the remaining quota of balls for different bowler types. This is a future value, but under the assumption of 5 bowlers or 4 bowlers with two part time bowlers, that future value is already available for the team. 

# Input --> Batter,  Team , Ball number, Innings, Non- striker  
# Ouput --> Win Probability

# In[1]:

import pickle
import pandas as pd

# Load the data
data = pd.read_csv('./Data/selected_data/processed_data.csv')
data.info()


# Selecting Player "KM Mendis"

# In[14]:


# sl_batsmen = data[data['batting_team'] == "Sri Lanka"]['batter'].unique()
# sl_batsmen


# In[2]:


all_batsmen = data['batter'].unique()
all_batsmen


# In[15]:


# SL ICC T20 Team
# selected_batters = ["PWH de Silva",'KIC Asalanka','BKG Mendis',"P Nissanka",'PHKD Mendis','S Samarawickrama','AD Mathews','MD Shanaka','DM de Silva','M Theekshana','PVD Chameera','N Thushara','M Pathirana','D Madushanka']


# In[4]:


def getPlayerScores(player_name: str, innings: list[int] = [1, 2] ) -> pd.DataFrame:
    # Get the data for BKG Mendis if batter is BKG Mendis or non-striker is BKG Mendis
	player_data = data.loc[
		((data['batter'] == player_name) | (data['non_striker'] == player_name)) & (data['innings'].isin(innings))
	]

	player_data.head()

	# 3 matches missing from the data
	# group data by match_id
	gp = player_data.groupby('match_id')
	cols = ['batting_team', 'batter', 'non_striker', 'batter_runs', 'balls_faced', 'wicket_type', 'won', 'innings', 'over', 'delivery', 'wickets_fallen','bowling_team','venue']
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


# In[17]:


# getPlayerScores('BKG Mendis')

# merged_df = pd.DataFrame()

# for player in selected_batters:
#     player_scores = getPlayerScores(player)
    
#     merged_df = pd.concat([merged_df, player_scores])



# In[5]:


merged_df = pd.DataFrame()

for player in all_batsmen:
    player_scores = getPlayerScores(player)
    
    merged_df = pd.concat([merged_df, player_scores])


# In[6]:


merged_df.info()


# In[7]:


merged_df.loc[merged_df["strike_rate"].isnull()]


# In[8]:


merged_df["strike_rate"] = merged_df["strike_rate"].fillna(0)


# In[9]:


merged_df.info()


# In[10]:


merged_df.head()


# In[11]:


merged_df.info()


# In[12]:


merged_df.drop(columns=["batter_runs","balls_faced","wicket_type","strike_rate","out","last_ball"],inplace=True)


# In[13]:


merged_df.info()


# In[14]:


merged_df.to_csv('./all_batters.csv',index=False)


# In[15]:


from sklearn.model_selection import cross_val_score, train_test_split # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import accuracy_score# type: ignore
RANDOM_STATE = 42
data =merged_df

# features = ['batter', 'innings', 'wickets_fallen', 'bowling_team', 'first_ball']
target = 'won'

# Preprocess the data
data = pd.get_dummies(data=data,columns=['batter', 'batting_team', 'bowling_team','venue'],dtype=int)



data.head()
data["won"].value_counts()
y = data[target]
X = data.drop(columns=[target])

model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
scores = cross_val_score(model, X, y, cv=5)

# Print the mean and standard deviation of the cross-validation scores
print('Cross-validation scores:', scores)
print('Mean cross-validation score:', scores.mean())
print('Standard deviation of cross-validation scores:', scores.std())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
win_probability = model.predict_proba(X_test)[:, 1]
print('Win probability:', win_probability)







# In[16]:


X_test.head()


# In[17]:


win_proba_df = pd.DataFrame(win_probability, columns=['win_probability'])

X_test = X_test.reset_index(drop=True)
win_proba_df = win_proba_df.reset_index(drop=True)
result = pd.concat([X_test, win_proba_df], axis=1)

result.head()


# In[18]:


y_train.value_counts()


# In[19]:


y_test.value_counts()

# Save the trained model
with open('random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)