import datetime

import numpy as np
import pandas as pd
import pytz as pytz
import yfinance as yf
from matplotlib import pyplot as plt


from model.preprocessors import process_inputs,process_targets



if __name__ == "__main__":
    # Download price histories from Yahoo Finance
    aapl = yf.Ticker("aapl")
    price_series = aapl.history(period='max')['Close'].dropna()
    x_df = process_inputs(price_series, window_length=10)
    y_series = process_targets(price_series)
    
    # Only keep rows in which we have both inputs and data.
    common_index = x_df.index.intersection(y_series.index)
    x_df, y_series = x_df.loc[common_index], y_series.loc[common_index]

    # Train and test model on a walk forward basis with a year gap inbetween
    for training_year in range(2007, datetime.date.today().year + 1 ):
        training_cutoff = datetime.datetime(training_year, 1, 1, tzinfo=pytz.timezone('America/New_York'))
        test_cutoff = datetime.datetime(training_year + 1, 1, 1, tzinfo=pytz.timezone('America/New_York'))

         # Isolate training data consisting of every data point before `training_year`
        training_x_series = x_df.loc[x_df.index < training_cutoff]
        training_y_series = y_series.loc[y_series.index < training_cutoff]



    
    



    