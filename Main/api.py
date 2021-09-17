from flask import Flask, Response, request, send_file
from flask_restful import Resource
import json
from Main.model import LSTM_Forecast
import pandas as pd

class ForecastApi(Resource):
    def get(self):
        body = request.get_json()
        data = body['data']
        days = body['days']
        date= body['dates']
        data_name = body['data_name']
        # convert data to list and then to pandas dataframe
        data = pd.DataFrame(
            {
                data_name: data,
                'date': date
            }
            )
        last = len(date)
        lstm = LSTM_Forecast(df = data, var = data_name, days=last)
        # make prediction
        # last = lenght of data
        pred = lstm.predict(last = last, days = days)
        # convert dataframe to list
        pred = pred.values.tolist()
        # convert list of list to list
        pred = [item for sublist in pred for item in sublist]
        data = {
            "predicted": pred
            }

        return Response(json.dumps(data), mimetype='application/json')



