import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance_matrix

file_path = "./data/nba_2425_stats.csv"
data = pd.read_csv(file_path)
# print("\nFirst few rows of data:")
# print(data.head())

# recommendations based on players stats' similarity

similarity_columns = ['MPG', 'PPG', 'RPG', 'APG', 'SPG', 'BPG', 'TPG']
similarity_data = data[similarity_columns]

scaler = MinMaxScaler()
normalized_similarity_data = scaler.fit_transform(similarity_data) # scale the data between 0 and 1
normalized_data = pd.DataFrame(normalized_similarity_data, columns=similarity_columns) # convert to dataframe
print(normalized_data.head())

# use distance matrix to see correlation. the closer to 0, the more similar the players are
dist_matrix = distance_matrix(normalized_data.values, normalized_data.values)

# convert to dataframe for better visualization
dist_df = pd.DataFrame(dist_matrix, index=data['NAME'], columns=data['NAME'])

# similarities of top 25 players
t25 = data.head(25)
t25.dist_df = dist_df.loc[t25['NAME'], t25['NAME']] # locates the top 25 players in the distance matrix
print(t25.dist_df)