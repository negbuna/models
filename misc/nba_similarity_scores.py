import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance_matrix

## -- NOT FINISHED --

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

# use euclidean distance to find similarity between players. it's like mutli-dimensional distance


# use distance matrix to see correlation. the closer to 0, the more similar the players are
dist_matrix = distance_matrix(normalized_data.values, normalized_data.values)

# convert to dataframe for better visualization
dist_df = pd.DataFrame(dist_matrix, index=data['NAME'], columns=data['NAME'])

# similarities of top 25 players
t25 = data.head(25)
t25.dist_df = dist_df.loc[t25['NAME'], t25['NAME']] # locates the top 25 players in the distance matrix
print(t25.dist_df)

print()
def trade_probability():
    # Sample the first random player
    random_row1 = data.sample(n=1) # returns n random rows
    random_player1 = random_row1['NAME'].values[0] # scalar value of the element in the random row, 'NAME' column 

    
    # Sample the second random player and ensure it's different from the first
    while True:
        random_row2 = data.sample(n=1)
        random_player2 = random_row2['NAME'].values[0]
        if random_player2 != random_player1:
            break
    
    print(f"Random player 1: {random_player1}")
    print(f"Random player 2: {random_player2}")
    
    # Find the euclidian distance (similarity) between the two players
    similarity = dist_df.loc[random_player1, random_player2]
    print(f"Similarity (Euclidean distance) between {random_player1} and {random_player2}: {similarity}")
    
    if similarity < 0.1:
        print(f"{random_player1} and {random_player2} are very similar and could probably be traded for each other.")
    elif 0.1 <= similarity < 0.3:
        print(f"{random_player1} and {random_player2} are somewhat similar and could potentially be traded for each other.")
    else:
        print(f"{random_player1} and {random_player2} are not similar and are unlikely to be traded for each other.")

trade_probability()