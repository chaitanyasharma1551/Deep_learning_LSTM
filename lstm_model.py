from numpy import concatenate
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, explained_variance_score,mean_absolute_error,r2_score, accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from matplotlib import pyplot
from math import sqrt
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	this function takes a lag value and creates a time series according to the lag value
	for ex. -
	lag=100
	timeseries= t(-99),t(-98),.........t(0)
	The deep learning lstm model will try to predict the t(1) value
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# load dataset
dataset = pd.read_csv('F:\\MS\\Cloudtrail\\lstminput.csv', header=0, index_col=0)
print(dataset.columns)
values = dataset.values
# integer encode direction
# encoder = LabelEncoder()
# values[:,4] = encoder.fit_transform(values[:,4])
# # ensure all data is float
# values = values.astype('float32')
# # normalize features
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled = scaler.fit_transform(values)
# # frame as supervised learning
error = []
lag = [100,150,200,300]
#The code takes different lag values and create a time series for every lag

for lagvalue in lag:
#error should be an array
	reframed = series_to_supervised(values, lagvalue, 1)
	print(reframed.columns)
# print(reframed)
# # drop columns we don't want to predict
	reframed.drop(columns=['var1(t)','var2(t)','var4(t)','var5(t)'], axis=1, inplace=True)
	print(reframed.columns)
	print(reframed.shape)
# split into train and test sets
	values = reframed.values
	print(values[:,-1])
	n_train_hours = 7008
	train = values[:n_train_hours, :]
	test = values[n_train_hours:, :]
# # split into input and outputs
	train_X, train_y = train[:, :-1], train[:, -1]
	test_X, test_y = test[:, :-1], test[:, -1]
# print(train_X.shape)
# # reshape input to be 3D [samples, timesteps, features]
	train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
	test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
	print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
	x=100                # we take dropout as a variable to find best dropout value
#vary the value of dropout
# # design network
	model = Sequential()
	model.add(LSTM(x, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
	model.add(Dropout(0.1))
	model.add(LSTM(units=x,return_sequences=True))
	model.add(Dropout(0.1))
	model.add(LSTM(units=x,return_sequences=True))
	model.add(Dropout(0.1))
	model.add(LSTM(units=x,return_sequences=True))
	model.add(Dropout(0.1))
	model.add(LSTM(units=x,return_sequences=True))
	model.add(Dropout(0.1))
	model.add(LSTM(units=x,return_sequences=False))
	model.add(Dropout(0.1))
	model.add(Dense(1))
	model.compile(loss='binary_crossentropy', optimizer='adam' , metrics=['accuracy'])
# # fit network
	history = model.fit(train_X, train_y, epochs=1, batch_size=100, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
# 	pyplot.plot(history.history['loss'], label='train')
# 	pyplot.plot(history.history['val_loss'], label='test')
# 	pyplot.plot(history.history['acc'])
# 	pyplot.plot(history.history['val_acc'])
# 	pyplot.title('model accuracy')
# 	pyplot.ylabel('accuracy')
# 	pyplot.xlabel('epoch')
# 	pyplot.legend(['train', 'test'], loc='upper left')
# 	pyplot.show()

#label x and y axis
#
	# pyplot.legend()
	# pyplot.show()
#
# make a prediction
	yhat = model.predict(test_X)


	test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# invert scaling for forecast
	inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
#more details required
# inv_yhat = scaler.inverse_transform(inv_yhat)
	inv_yhat = inv_yhat[:,0]
	print(type(inv_yhat))
# invert scaling for actual
	test_y = test_y.reshape((len(test_y), 1))
	inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
	print(type(inv_y))
# inv_y = scaler.inverse_transform(inv_y)
	inv_y = inv_y[:,0]
# calculate RMSE
	maap = np.mean(np.arctan(np.float64(np.abs((inv_y - inv_yhat)) / inv_y))) * 100
	print('Test MAAP: %.3f'%maap)
	acc = r2_score(inv_y,inv_yhat)*100
	print('Test Accuracy: %.3f' % acc)
	rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
	print('Test RMSE: %.3f' % rmse)
	mae = mean_absolute_error(inv_y,inv_yhat)
	print("Test MAE: %.3f" %mae)
	evs = explained_variance_score(inv_y,inv_yhat)
	print("Test Explained Variance Score: %.3f" %evs)
	# error.append(rmse)


