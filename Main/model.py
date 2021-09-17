from keras.backend import var
import streamlit as st
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import numpy as np
import pandas as pd

class LSTM_Forecast:
    def __init__(self,df,var,days):
        self.df = df
        self.dataset = None
        self.x_train = None
        self.y_train = None
        self.var =  var
        self.days = 10
        self.data = []
        self.re = None

    def data_preprocessing(self):
        scalar = StandardScaler()
        scaled_data = scalar.fit_transform(self.dataset)
        return scaled_data

    def split(self):
        self.x_train, self.y_train = [],[]
        for i in range(self.days, len(self.dataset), self.days):
            self.x_train.append(self.dataset[i-self.days:i, 0])
            self.y_train.append(self.dataset[i, 0])
        self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)
        self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], self.x_train.shape[1], 1))



    def Build(self):
        data = self.df.filter([self.var])
        self.dataset = data.values
        self.dataset = self.data_preprocessing()
        self.split()
        # self.df["returns"] = self.df[self.var].pct_change() 
        # self.df["log_returns"] = np.log(1+self.df["returns"])
        # self.logret = self.df[["Date","log_returns"]]
        # self.var = "log_returns"

        model = Sequential()
        model.add(LSTM(10, return_sequences=True, input_shape= (self.x_train.shape[1], 1)))
        model.add(LSTM(10, return_sequences= False))
        model.add(Dense(10))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(self.x_train, self.y_train, batch_size=2, epochs=30)
        return model

    def predict(self, last = 10, days = 0 ):
        self.data = self.df.filter([self.var])
        for i in range(days):
            ndays = int(last)
            dataset = self.data.values
            dataset = dataset[-1:-(ndays+1):-1]
            dataset = dataset.reshape(-1, 1)
            model = self.Build()
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(dataset)
            x_pred = np.reshape(scaled_data, (1, scaled_data.shape[0], 1 ))
            pred = model.predict(x_pred)
            pred = scaler.inverse_transform(pred)
            # append pred[0][0] to self.re dataframe
            self.data.loc[len(self.data)] = [pred[0][0]]

            #self.data.append(pred[0][0])

        return self.data