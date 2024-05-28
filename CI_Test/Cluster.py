# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the CSV file
file_path = os.path.join(script_dir, '../Data/selected_data/processed_data.csv')

# Read the CSV file
data = pd.read_csv(file_path)

# %% [markdown]
# ### **Batsman Clustering**

# %%
unique_batters = data['batter'].unique()

for batter in unique_batters:
    print(batter)

# %%
# Average score per match of a batsman
batsman_match_runs = data.groupby(['batter', 'match_id'])['runs_by_bat'].sum().reset_index()

batsman_avg_score = batsman_match_runs.groupby('batter')['runs_by_bat'].mean().reset_index()
batsman_avg_score.rename(columns={'runs_by_bat': 'average_score_per_match'}, inplace=True)

print(batsman_avg_score)

# %%
# Maximum number of balls faced in each match
max_balls_faced = data.groupby(['batter', 'match_id'])['balls_faced'].max().reset_index()
print("max_balls_faced")
print(max_balls_faced)

# %%
# Sum up the maximum balls faced in each match
total_balls_faced = max_balls_faced.groupby('batter')['balls_faced'].sum().reset_index()

print(total_balls_faced)

# %%
# Calculate the total runs scored by each batsman
batsman_total_runs = data.groupby('batter')['runs_by_bat'].sum().reset_index()

print(batsman_total_runs)

# %%
# Merge batsman_total_runs with max_balls_faced on the 'batter' column
batsman_strike_rate = pd.merge(batsman_total_runs, total_balls_faced, on='batter')

# Calculate strike rate
batsman_strike_rate['strike_rate'] = (batsman_strike_rate['runs_by_bat'] / batsman_strike_rate['balls_faced']) * 100

print(batsman_strike_rate)

# %%
# Merge the two metrics into a single DataFrame
batsman_metrics = pd.merge(batsman_avg_score, batsman_strike_rate[['batter', 'strike_rate']], on='batter')

print("Batsman Metrics before Clustering:")
print(batsman_metrics)

# %%
plt.figure(figsize=(8, 5))
sns.scatterplot(data=batsman_metrics, x='average_score_per_match', y='strike_rate')
plt.title('Scatter Plot of Batsmen Strike Rate vs Average Score')
plt.xlabel('Average Score per Match')
plt.ylabel('Strike Rate')
# plt.show()

# %%
# Elbow Method
sse = []  # Sum of squared distances to the closest cluster center
k_range = range(1, 11)

k_range = range(1, 11)  # Range of k values to try
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(batsman_metrics[['average_score_per_match', 'strike_rate']])
    sse.append(kmeans.inertia_)

# %%
# Plot the Elbow Method results
plt.figure(figsize=(10, 6))
plt.plot(k_range, sse, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared Distances')
# plt.show()

# %%
print("sse: ",sse)

# %%
# k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
batsman_metrics['batsman_cluster'] = kmeans.fit_predict(batsman_metrics[['average_score_per_match', 'strike_rate']])

# %%
print("Batsman Metrics after Clustering:")
print(batsman_metrics.head())

# %%
plt.figure(figsize=(10, 6))
sns.scatterplot(data=batsman_metrics, x='average_score_per_match', y='strike_rate', hue='batsman_cluster', palette='viridis')
plt.title('K-means Clustering of Batsmen based on Strike Rate and Average Score')
plt.xlabel('Average Score per Match')
plt.ylabel('Strike Rate')
plt.legend(title='Cluster')
# plt.show()
plt.savefig("Clustering_of_Batsmen.png",dpi=120) 
plt.close()

# %% [markdown]
# ### **Bowler Clustering**

# %%
# Calculate the average wickets per match for each bowler
bowler_wickets = data[data['wicket_type'].notnull()]

bowler_match_wickets = bowler_wickets.groupby(['bowler', 'match_id']).size().reset_index(name='wickets_taken')

bowler_avg_wickets = bowler_match_wickets.groupby('bowler')['wickets_taken'].mean().reset_index()
bowler_avg_wickets.rename(columns={'wickets_taken': 'average_wickets_per_match'}, inplace=True)

# %%
print(bowler_avg_wickets)

# %%
# Calculate economy rate for each bowler
bowler_stats = data.groupby('bowler').agg(
    total_runs_conceded=pd.NamedAgg(column='total_runs_delivery', aggfunc='sum'),
    total_overs_bowled=pd.NamedAgg(column='over', aggfunc='nunique')
).reset_index()
bowler_stats['economy_rate'] = (bowler_stats['total_runs_conceded'] / bowler_stats['total_overs_bowled'])

# %%
print(bowler_stats)

# %%
# Merge the metrics into a single DataFrame
bowler_metrics = pd.merge(bowler_avg_wickets, bowler_stats[['bowler', 'economy_rate']], on='bowler')
print("Bowler Metrics before Clustering:")
print(bowler_metrics)

# %%
plt.figure(figsize=(10, 6))
sns.scatterplot(data=bowler_metrics, x='average_wickets_per_match', y='economy_rate')
plt.title('Scatter Plot of Bowler Economy Rate vs Average Wickets')
plt.xlabel('Average Wickets per Match')
plt.ylabel('Economy Rate')
# plt.show()

# %%
#Elbow method
sse = []  # Sum of squared distances to the closest cluster center
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(bowler_metrics[['average_wickets_per_match', 'economy_rate']])
    sse.append(kmeans.inertia_)

# Plot the Elbow Method results
plt.figure(figsize=(10, 6))
plt.plot(k_range, sse, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared Distances (SSE)')
# plt.show()

# %%
print("sse: ",sse)

# %%
# Cluster bowlers using K-means
kmeans = KMeans(n_clusters=3, random_state=42)  # Choose the number of clusters
bowler_metrics['bowler_cluster'] = kmeans.fit_predict(bowler_metrics[['average_wickets_per_match', 'economy_rate']])

print("Bowler Metrics after Clustering:")
print(bowler_metrics.head())

# %%
# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=bowler_metrics, x='average_wickets_per_match', y='economy_rate', hue='bowler_cluster', palette='viridis')
plt.title('K-means Clustering of Bowlers based on Average Wickets per Match and Economy Rate')
plt.xlabel('Average Wickets per Match')
plt.ylabel('Economy Rate')
plt.legend(title='Cluster')
# plt.show()
plt.savefig("Clustering_of_Bowlers.png",dpi=120) 
plt.close()

# %% [markdown]
# ### **Chi Square Statistics**

# %%
from scipy.stats import chi2_contingency

# %%
# Merge batsman and bowler metrics on common players
merged_metrics = pd.merge(batsman_metrics, bowler_metrics, left_on='batter', right_on='bowler')

# Create a contingency table
contingency_table = pd.crosstab(merged_metrics['batsman_cluster'], merged_metrics['bowler_cluster'])

# Perform the chi-square test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-square statistic: {chi2}")
print(f"P-value: {p}")

# Write scores to a file
with open("metrics.txt", 'w') as outfile:
        outfile.write("Chi-square statistic:: %2.5f%%\n" % chi2)
        outfile.write("P-value: %2.5f%%\n" % p)

# %%
# Plot heatmap of the contingency table
plt.figure(figsize=(5, 4))
sns.heatmap(contingency_table, annot=True, cmap="YlGnBu", fmt="d")
plt.title('Contingency Table: Batsman Clusters vs Bowler Clusters')
plt.xlabel('Bowler Clusters')
plt.ylabel('Batsman Clusters')
# plt.show()
plt.savefig("Contingency_Table.png",dpi=120) 
plt.close()

# %%
# Batsman Metrics: Describe cluster definitions
batsman_cluster_description = batsman_metrics.groupby('batsman_cluster').agg(
    average_score_per_match_mean=('average_score_per_match', 'mean'),
    strike_rate_mean=('strike_rate', 'mean')
)
print("Batsman Cluster Definitions:")
print(batsman_cluster_description)

# Bowler Metrics: Describe cluster definitions
bowler_cluster_description = bowler_metrics.groupby('bowler_cluster').agg(
    average_wickets_per_match_mean=('average_wickets_per_match', 'mean'),
    economy_rate_mean=('economy_rate', 'mean')
)
print("\nBowler Cluster Definitions:")
print(bowler_cluster_description)