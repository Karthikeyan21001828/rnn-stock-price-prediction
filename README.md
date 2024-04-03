# Stock Price Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset
Develop a system for predicting stock prices using historical data. Implement an algorithm that preprocesses training and test datasets, trains a Bidirectional SimpleRNN model on the training data, and generates predictions for the test data. The model should be capable of handling sequences of stock prices with a length of 60, and the predictions should be scaled back to their original form. Visualize the real and predicted stock prices using matplotlib. Additionally, include author information in the output.

![image](https://github.com/Karthikeyan21001828/rnn-stock-price-prediction/assets/93427303/d1a471ae-90d1-4179-9f86-b4092a82d83a)

## Design Steps

### Step 1:
Import numpy, matplotlib, pandas, MinMaxScaler from sklearn, layers and Sequential from keras.
### Step 2:
Load training dataset using pandas, extract relevant columns, perform Min-Max scaling.
### Step 3:
Create input-output pairs with a sequence length of 60, reshape input data to fit the RNN model.
### Step 4:
Initialize a Sequential model, add a Bidirectional SimpleRNN layer with 60 units, add a Dense layer, compile the model with 'adam' optimizer and mean squared error loss.
### Step 5:
Fit the model to the training data for 100 epochs with a batch size of 32.
### Step 6:
Load test dataset using pandas, extract relevant columns, concatenate datasets, transform data using MinMaxScaler.
### Step 7:
Create input sequences for test data with a sequence length of 60, reshape test data to fit the model.
### Step 8:
Use the trained model to predict stock prices for test data, inverse transform predicted prices to original scale.
### Step 9:
Plot real and predicted stock prices using matplotlib, show plot with appropriate labels and legends.
### Step 10:
 Print the name and register number of the author.

## Program
#### Name: Karthikeyan K
#### Register Number: 212221230046

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential
dataset_train = pd.read_csv('trainset.csv')
dataset_train.columns
train_set = dataset_train.iloc[:,1:2].values
type(train_set)
train_set.shape
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)
training_set_scaled.shape
X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))
X_train.shape
length = 60
n_features = 1
model = Sequential()
model.add(layers.Bidirectional(layers.SimpleRNN(units=60),input_shape=(length,n_features)))
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mse')
print("Name: KARTHIKEYAN K\nRegister Number: 212221230046")
model.summary()
model.fit(X_train1,y_train,epochs=100, batch_size=32)
dataset_test = pd.read_csv('testset.csv')
test_set = dataset_test.iloc[:,1:2].values
test_set.shape
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
for i in range(60,1384):
  X_test.append(inputs_scaled[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))
X_test.shape
predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)
print("Name: KARTHIKEYAN K\nRegister Number: 212221230046")
plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='blue', label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
```

## Output

### True Stock Price, Predicted Stock Price vs time

![image](https://github.com/Karthikeyan21001828/rnn-stock-price-prediction/assets/93427303/10d2c10a-f0c5-47ec-be23-8c683b4e281f)


### Mean Square Error

![image](https://github.com/Karthikeyan21001828/rnn-stock-price-prediction/assets/93427303/84505020-1095-4dda-9018-f9ec67727091)


## Result
A Recurrent Neural Network model for stock price prediction has been developed and executed successfully.
