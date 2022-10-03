import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Dense, Dropout, LSTM
from tensorflow.python.keras.models import Sequential

company = 'FB'
start = dt.datetime(2016,1,1)
end = dt.datetime(2021,1,1)
#Acquire Data
data = web.DataReader(company, 'yahoo', start, end)
adjdata = data['Close'].values
dates =[]
for x in range(len(data)):
    newdate = str(data.index[x])
    newdate = newdate[0:10]
    dates.append(newdate)
data['dates'] = dates
# dates = np.array(x_train)[indices.astype(int)]
doublediff = np.diff(np.sign(np.diff(adjdata)))
peak_locations = np.where(doublediff == -2)[0] + 1
doublediff2 = np.diff(np.sign(np.diff(-1*adjdata)))
trough_locations = np.where(doublediff2 == -2)[0] + 1
#Prepare data using closing prices
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

prediction_days = 50
future_day = 30
x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data) - future_day):
	x_train.append(scaled_data[x-prediction_days:x, 0])
	y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#Build the Model
model = Sequential()
model.add(LSTM(units=60, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2)) # In neural nets dropout prevents overfit by randomly removing weights to prevent memorization of trends
model.add(LSTM(units=60, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=60))
model.add(Dropout(0.2))
model.add(Dense(units=1)) # Predicts next closing price
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=20, batch_size=30) # Model sees same data 20 times / 30 units at once
#Test accuracy on existing data
test_start = dt.datetime(2021,1,1)
test_end = dt.datetime.now()
test_data = web.DataReader(company, 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values
total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)
#Predict Test Data
x_test = []
for x in range(prediction_days, len(model_inputs)) :
	x_test.append(model_inputs[x-prediction_days:x, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
#Plot Test Predictions
"""
plt.plot(actual_prices, color="black", label=f"Actual {company} Price")
"""
plt.figure(figsize=(16,10), dpi= 80)
plt.plot(predicted_prices, color='red', label=f"Predicted {company} Price")
plt.plot('dates', 'Close', adjdata=data, color='tab:blue', label='Closing Price(USD)')
plt.scatter(dates[peak_locations], adjdata.Close[peak_locations], marker=plt.markers.CARETUPBASE, color='tab:green', s=100, label='Peaks')
plt.scatter(dates[trough_locations], adjdata.Close[trough_locations], marker=plt.markers.CARETDOWNBASE, color='tab:red', s=100, label='Troughs')
for t, p in zip(trough_locations[1::5], peak_locations[::3]):
    plt.text(dates[p], adjdata.Close[p]+15, dates[p], horizontalalignment='center', color='darkgreen')
    plt.text(dates[t], adjdata.Close[t]-35, dates[t], horizontalalignment='center', color='darkred')
plt.ylim(50,750)
xtick_location = adjdata.index.tolist()[::6]
xtick_labels = adjdata.date.tolist()[::6]
plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=90, fontsize=12, alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.3)
plt.grid(axis='y', alpha=.3)
plt.title(f"{company} Share Price Over Time")
plt.xlabel('Time')
# plt.ylabel(f'{company} Share Price($USD)')
plt.legend()
plt.show()
#Predict next day
real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs+1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1],1))
prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")