# Author: Helal Chowdhury
# Version: 1

from __future__ import print_function, division
import pandas as pd
import math
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from dataclasses import dataclass

import mpld3
import streamlit.components.v1 as components

#--------------------------------

st.title("Interactive Timeseries Analysis")
#st.subheader("Image is predicted as {}".format(klas[predicted.item()])   )
                                                                  
#-------------------------------------------------------------------------------------------
def plot_series(time, series, format="-", start=0, end=None):
    """Helper function to plot our time series"""
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(False)

def trend(time, slope=0):
    """Define the trend through slope and time"""
    return slope * time

def seasonal_pattern(season_time):
    """Arbitrary definition of a seasonality pattern"""
    return np.where(season_time < 0.1,
                    np.cos(season_time * 6 * np.pi),
                    2 / np.exp(9 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats a pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    """Adds white noise to the series"""
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level
    


def generate_time_series():
    # Temporal dimension: 4 years of data
    time = np.arange(4 * 365 + 1, dtype="float32")

    # The initial series is nothing more than a straight line which we will then modify with the other functions
    y_intercept = 10
    slope = 0.005
    series = trend(time, slope) + y_intercept

    # Add seasonality
    amplitude = 50
    series += seasonality(time, period=365, amplitude=amplitude)

    # Add noise
    noise_level = 3
    series += noise(time, noise_level, seed=51)
    
    return time, series

# Let's save the parameters of our time series in the dataclass
@dataclass
class G:
    TIME, SERIES = generate_time_series()
    SPLIT_TIME = 1100 # on day 1100 the training period will end. The rest will belong to the validation set
    WINDOW_SIZE = 20 # how many data points will we take into account to make our prediction
    BATCH_SIZE = 32 # how many items will we supply per batch
    SHUFFLE_BUFFER_SIZE = 1000 # we need this parameter to define the Tensorflow sample buffer   


def create_model():
  # define a sequential model
  model = tf.keras.models.Sequential([ 
      tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                    input_shape=[None]),
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1024, return_sequences=True)),
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True)),
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
      tf.keras.layers.Dense(1),
  ]) 

  return model
#------------------
@st.cache(allow_output_mutation=True)
def load_model():
  model1=create_model()
  model1.load_weights("nn_weights.h5")
  return model1


#---------------streamlit-----------

number1 = st.number_input('Choose Initial Sample index',0)
#st.write('The current number1 is ', number1)

number2 = st.number_input('Choose Final Sample index',1)
#st.write('The current number2 is ', number2)
#number1=int(number1)
#number2=int(number2)
init=st.slider("Initial Sample index:",number1,number2 )    
final=st.slider("Final Sample index:",number1 ,number2 )  
dif=final-init+1
st.write("Number of Samples Choosen:",dif)
st.info("Performance of Bi-directional LSTM time series forecasting: Minimum mean square erro, Minimum absolute error")
st.info('Model is trained with window size=20', icon="ℹ️")

G_TIME=G.TIME[init:final+1]

G_SERIES=G.SERIES[init:final+1]

G_MAX=max(G_TIME)
new_forecast_series = np.expand_dims(G_SERIES, axis=0)
model1=load_model()

pred =model1.predict(new_forecast_series)
true=G.SERIES[final+1]

mse = tf.keras.metrics.mean_squared_error(true, pred).numpy()
mae = tf.keras.metrics.mean_absolute_error(true, pred).numpy()
data={"MSE":mse,"MAE":mae}
df=pd.DataFrame(data)
st.dataframe(df)

plot_figure=plt.figure(figsize=(20, 10))
plt.plot(G_TIME, G_SERIES, label="Selected time series")
plt.scatter(G_MAX+1, pred, color="red", marker="x", s=70, label="prediction")
plt.scatter(G_MAX+1, G.SERIES[final+1], color="blue", marker="o", s=70, label="Original")
plt.legend()
st.pyplot(plot_figure, label="last 100 points of time series")

####################

fig = plt.figure() 
plt.scatter(G_TIME, G_SERIES,marker=".", s=30, label="Selected time series") 
plt.scatter(G_MAX+1, pred, color="red", marker="x", s=70, label="prediction")
plt.scatter(G_MAX+1, G.SERIES[final+1], color="blue", marker="o", s=70, label="Original")
plt.legend()
fig_html = mpld3.fig_to_html(fig)
components.html(fig_html, width=800,height=600)







