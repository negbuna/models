import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import os

# -- NOT FINISHED --

red_file_path = "./data/winequalityred.csv"
white_file_path = "./data/winequalitywhite.csv"
print(os.path.exists(red_file_path))

red_df = pd.read_csv(red_file_path, sep=';')

# print()
# print(red_df.isnull().any()) # no missing data
print()
print(red_df.head(30))

# rating > 7 - good quality
# 5 <= rating <= 7 - average quality
# rating < 5 - poor quality

num_wines_red = red_df.shape[0] # rows in the dataframe
num_good_red = red_df[red_df['quality'] > 7].shape[0]
num_average_red = red_df[(red_df['quality'] >= 5) & (red_df['quality'] <= 7)].shape[0]
num_poor_red = red_df[red_df['quality'] < 5].shape[0]
percentage_good_red = (num_good_red / num_wines_red) * 100

print()
print(f"Number of red wines: {num_wines_red}")
print(f"Number of good red wines: {num_good_red}")
print(f"Number of average red wines: {num_average_red}")
print(f"Number of poor red wines: {num_poor_red}")
print(f"Percentage of good red wines: {percentage_good_red:.2f}%")

# distribution plots
plt.figure(figsize=(13, 6.5))  # 14 inches wide, 7 inches tall

# Histogram
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
# bins = number of bars, color = color of bars, alpha = transparency
plt.hist(red_df['quality'], bins=7, color='red', alpha=0.7)
plt.title('Distribution of Wine Quality')
plt.xlabel('Quality Score')
plt.ylabel('Number of Wines')

# Box plot
plt.subplot(1, 2, 2)
# vert = vertical, patch_artist = fill color, 
# boxprops = box properties, facecolor = inside color, color = border color
plt.boxplot(red_df['quality'], vert=False, patch_artist=True, boxprops=dict(facecolor='red', color='red'))
plt.title('Box Plot of Wine Quality')
plt.xlabel('Quality Score')

plt.tight_layout()
plt.show()