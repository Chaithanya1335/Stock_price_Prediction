import streamlit as st
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error


st.set_page_config("Stock Price Prediction")

st.header("Stock Price Prediction")
# Load the data
data = pd.read_csv('AAPL.csv')

st.sidebar.selectbox("Select which Need to predicted",options=['close','high','low','open','volume'])
# Select the column to predict
df = data.reset_index()['close']

# Scale the data
scaler = MinMaxScaler()
df = scaler.fit_transform(np.array(df).reshape(-1, 1))

# Split the data into training and test sets
train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
train_data, test_data = df[:train_size, :], df[train_size:len(df), :1]

# Function to create dataset in time step format
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# Prepare the datasets
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

# Reshape the input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

with st.spinner("Predicting Please wait...."):
    # Train the model
    model.fit(X_train, y_train, validation_data=(X_test, ytest), epochs=100, batch_size=64, verbose=1)

    # Predictions and performance metrics
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Transform back to original form
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    # Calculate RMSE
    print(f"Training RMSE: {math.sqrt(mean_squared_error(y_train, train_predict))}")
    print(f"Test RMSE: {math.sqrt(mean_squared_error(ytest, test_predict))}")

    # Preparing input for future predictions
    x_input = test_data[len(test_data) - 100:].reshape(1, -1)

    temp_input = list(x_input)
    temp_input=temp_input[0].tolist()
    lst_output = []
    n_steps = 100

    no_of_days = st.number_input("Enter no of days to Predict:",min_value=10,placeholder="Enter no of days to Predict")

    # Predicting next 10 days
    for i in range(no_of_days):
        if len(temp_input) > 100:
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            lst_output.extend(yhat.tolist())
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())

    # Visualize the predictions
    day_new = np.arange(1, 101)
    day_pred = np.arange(101, no_of_days+101)

  

    plt.title(f"New Predictions for Next {no_of_days} Days")
    plt.plot(day_new, scaler.inverse_transform(df[len(df) - 100:]).reshape(100,1))
    plt.plot(day_pred, scaler.inverse_transform(lst_output))
    plt.savefig("New_Predictions.png")

   
st.image("New_Predictions.png", caption="Predicted Stock Prices")


