# @title
"""
Chicago Taxi Trip Duration Prediction Model
A machine learning model to predict taxi trip durations and find relationships between other data using linear regression.
"""

# General
import io

# Data
import numpy as np
import pandas as pd

# ML
import keras

# Visualization
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()  # Turn on interactive mode at start

chicago_taxi_dataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")

# Only use specific columns
training_df = chicago_taxi_dataset[['TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE']]

print('Read dataset completed successfully.')

# Format the placeholder as the amt of elements in any index = rows
# print('Total number of rows: {0}\n\n'.format(len(training_df.index)))

# .head(n) shows the first n rows, defaults to 5
# training_df.head(200)

print(training_df.describe(include='all'))

max_fare = training_df['FARE'].max()
print(f"What is the maximum fare?: {max_fare.round(2)}")

mean_distance = training_df['TRIP_MILES'].mean()
print(f"What is the average trip distance?: {mean_distance.round(4)}")

num_companies = training_df['COMPANY'].nunique()
print(f"How many companies are in the data?: {num_companies}")

payment_type = training_df['PAYMENT_TYPE'].value_counts().idxmax() 
print(f"What is the most frequent payment type?: {payment_type}")
# .value_counts().idxmax() returns the SINGLE most frequent value, .mode() can return multiple

def find_missing_data(df):
    # .isnull().values.any() returns True if there are any missing values, faster for large datasets where just yes/no is needed
    # .isnull().sum().sum() returns the total number of missing values (integer), checks every single value in dataset
    missing_data = df.isnull().values.any()
    if missing_data:
        return "Yes"
    else:
        return "No"
    
print(f"Are any features missing data?: {find_missing_data(training_df)}")

# Correlation matrix
correlation_matrix = training_df.corr(numeric_only=True)
print(correlation_matrix)

# Pair plot - grid visualing the relationship of each feature with all other features
training_df_pairplot = sns.pairplot(
    training_df, 
    x_vars=["FARE", "TRIP_MILES", "TRIP_SECONDS"], # left to right
    y_vars=["FARE", "TRIP_MILES", "TRIP_SECONDS"] # top to bottom
)

# Making the model
def build_model(learning_rate, num_features):
    # Describe the topography of the model.
    # The topography of a simple linear regression model
    # is a single node in a single layer.
    inputs = keras.Input(shape=(num_features, )) # features are predicting the amount of columns
    outputs = keras.layers.Dense(units=1)(inputs) # units=1 specifies a single node, (inputs) applies the layer to the input data
    model = keras.Model(inputs=inputs, outputs=outputs) 

    # Model topography -> code that keras can efficiently execute. Configure to minimize model's mse.
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
                loss="mean_squared_error",
                metrics=[keras.metrics.RootMeanSquaredError()])

    return model

def train_model(model, df, features, label, epochs, batch_size):
    # Feed the model the feature and the label. It will train for the specified number of epochs.
    # input_x = df.iloc[:,1:3].values
    # df[feature]
    history = model.fit(
        x=features, # Input data (like trip_miles)
        y=label, # target data (like fare)
        batch_size=batch_size, # how many samples to process at once
        epochs=epochs # how many times to go through all the data
    )
  
    # Gather the trained model's weight and bias
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]

    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch

    # Isolate the error for each epoch.
    hist = pd.DataFrame(history.history)

    # snapshot of the model's rmse at each epoch.
    rmse = hist["root_mean_squared_error"]

    return trained_weight, trained_bias, epochs, rmse

def model_info(feature_names, label_name, model_output):
    """Creates a summary string about the model's training"""
    trained_weight, trained_bias, epochs, rmse = model_output
    
    info = "Model training summary:\n"
    info += f"Features used: {feature_names}\n"
    info += f"Label predicted: {label_name}\n"
    info += f"Final RMSE: {rmse.iloc[-1]:.4f}\n" # last value in the list
    info += f"Final bias: {trained_bias[0]:.4f}\n"
    info += f"Final weights: {[f'{w[0]:.4f}' for w in trained_weight]}\n"
    
    return info

def make_plots(df, feature_names, label_name, model_output):
    """Creates training progress and prediction plots"""
    trained_weight, trained_bias, epochs, rmse = model_output
    
    # Create new figure with unique number for each plot
    plt.figure(figsize=(10, 6))
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")
    plt.plot(epochs, rmse, label=f"RMSE for {feature_names}")  # Add feature names to label
    plt.legend()
    plt.draw()  # Draw but don't block
    plt.pause(0.1)  # Small pause to ensure plot shows up

def run_experiment(df, feature_names, label_name, learning_rate, epochs, batch_size):

    print('INFO: starting training experiment with features={} and label={}\n'.format(feature_names, label_name))

    num_features = len(feature_names)

    features = df.loc[:, feature_names].values
    label = df[label_name].values

    model = build_model(learning_rate, num_features)
    model_output = train_model(model, df, features, label, epochs, batch_size)

    print('\nSUCCESS: training experiment complete\n')
    print('{}'.format(model_info(feature_names, label_name, model_output)))
    make_plots(df, feature_names, label_name, model_output)

    return model

print("SUCCESS: defining linear regression functions complete.")

# Experiment 1

# These variables are the hyperparameters.
learning_rate = 0.001
epochs = 20
batch_size = 50

# Specify the feature and the label.
features = ['TRIP_MILES']
label = 'FARE'

model_1 = run_experiment(training_df, features, label, learning_rate, epochs, batch_size)

training_df.loc[:, 'TRIP_MINUTES'] = training_df['TRIP_SECONDS']/60 # : is used to select all rows

features2 = ['TRIP_MILES', 'TRIP_MINUTES']
label = 'FARE'

model_2 = run_experiment(training_df, features2, label, learning_rate, epochs, batch_size)

# Make predictions with model_1 (using just miles)
test_input_1 = np.array([[5.0]])  # Shape: (1 sample, 1 feature)
prediction_1 = model_1.predict(test_input_1)
print(f"\nModel 1 Prediction:")
print(f"Predicted fare for a 5-mile trip: ${prediction_1[0][0]:.2f}")

# Make predictions with model_2 (using miles and minutes)
test_input_2 = np.array([[5.0, 15.0]])  # Shape: (1 sample, 2 features)
prediction_2 = model_2.predict(test_input_2)
print(f"\nModel 2 Prediction:")
print(f"Predicted fare for a 5-mile, 15-minute trip: ${prediction_2[0][0]:.2f}")

# Make predictions with model_3
test_input_3 = np.array([[23.0, 42.0]])  # Shape: (1 sample, 3 features)
prediction_3 = model_2.predict(test_input_3)
tipped_fare = 1.15 * prediction_3[0][0]
print(f"\nModel 3 Prediction:")
print(f"Predicted fare for a 23-mile, 42-minute trip with 15% tip: ${tipped_fare:.2f}")

plt.ioff()  # Turn off interactive mode
plt.show(block=True)  # Keep all plots open until manually closed
