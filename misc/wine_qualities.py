import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
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

# axis = 0 -> apply drop to rows, axis = 1 -> apply drop to columns
features = red_df.drop('quality', axis=1) 
labels = red_df['quality']
print()
print(features)

# splitting data
# tuple unpacking: train_test_split takes features, labels, test_size (decimal -> percentage of data to be test data), random_state (seed)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
# y values aren't scaled since those are the ones i am trying to predict.
scalar = MinMaxScaler()

X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)

model = LinearRegression()
# training the model with the existing data
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled) # predicting y values based on the test data
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}") 

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='red') # scatterplot of actual vs predicted values

# plotting reference line
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='blue', linestyle="--", label="Reference (Ideal) Line") # line of best fit
# first list is range of x values for the reference line, second list is range y values for the reference line.
# reference line shows how far the models predictions are from being perfect
# - = solid line, -- = dashed line, -. = dash-dot line, : = dotted line

# plotting regression line
# sorted_indices = np.argsort(y_test) # sort the indices of y_test, do [::-1] for descending order
# sorted_y_test = y_test.iloc[sorted_indices] # locate the sorted y_test values
# sorted_y_pred = y_pred[sorted_indices] # retrieve sorted predicted y_pred values 

# plt.plot(sorted_y_test, sorted_y_pred, color='green', label="Regression Line") # line plot of actual vs predicted values

# labels and title
plt.xlabel('Actual Quality')    
plt.ylabel('Predicted Quality')
plt.title('Actual Quality vs Predicted Quality')
plt.show()

