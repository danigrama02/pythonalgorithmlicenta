import datetime
import matplotlib.pyplot as matplot

from pymongo import MongoClient
from keras.src.models import Sequential
from keras.src.layers import Dense, LSTM
import numpy as np
import tensorflow as tf
import pickle
from dateutil.parser import parse
import random
import functionsWrappup
def setup_model(nr_steps, nr_features):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True,activation='sigmoid', input_shape=(nr_steps, nr_features)))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(8))
    model.compile(optimizer='adam', loss='mse')
    return model
def prepare_data(data):
    prepared_data = []
    for el in data:
        row = []
        # row.append(float(el['date'].timestamp()))
        # row.append(float(el['lat']))
        # row.append(float(el['lng']))
        row.append(float(el['temperature_2m']))
        row.append(float(el['relative_humidity_2m']))
        row.append(float(el['precipitation']))
        row.append(float(el['rain']))
        row.append(float(el['showers']))
        row.append(float(el['snowfall']))
        row.append(float(el['weather_code']))
        row.append(float(el['surface_pressure']))
        prepared_data.append(row)
    return prepared_data
def train(data,model):
    x_tarin = []
    y_train = []
    x_validate =[]
    y_validate = []
    data.sort(reverse=False, key=lambda x: x[0])
    n = len(data)
    # for i in range(0,n,2):
    #     if i < n*0.8:
    #         x_tarin.append(data[i])
    #         y_train.append(data[i])
    #     else:
    #         x_validate.append(data[i])
    #         y_validate.append(data[i+1])
    i = 0
    n= len(data)
    for el in data:
        if i < n*0.8:
            x_tarin.append(el)
            y_train.append(el)
        else:
            x_validate.append(el)
            y_validate.append(el)
        i+=1
    x_train = np.array(x_tarin)
    y_train = np.array(y_train)

    x_validate = np.array(x_validate)
    y_validate = np.array(y_validate)
    print(x_train.shape)
    print(y_train.shape)
    print(x_validate.shape)
    print(y_validate.shape)
    x_train = x_train.reshape((x_train.shape[0], 1, 8))
    y_train = y_train.reshape((y_train.shape[0], 1, 8))
    x_validate = x_validate.reshape((x_validate.shape[0], 1, 8))
    y_validate = y_validate.reshape((y_validate.shape[0], 1, 8))
    model.fit(x_train, y_train, epochs=100, batch_size=1024, verbose=1, shuffle=False)
    model.evaluate(x_validate, y_validate, verbose=1)

    # x_train = x_train.reshape((x_train.shape[0], 1, num_features))
    # #y_train = y_train.reshape((y_train.shape[0], 1, y_train.shape[1]))
    # x_validate = x_validate.reshape((x_validate.shape[0], 1, num_features))
    # train_dataset = tf.data.Dataset.from_tensor_slices((x_train, x_validate))
    # print(len(train_dataset))
    # nb_train_steps = math.floor(len(train_dataset) / 32)
    # #Repeat the dataset for a number of epochs
    # #train_dataset = train_dataset.batch(32).repeat(100)
    # print(len(train_dataset))


    #print(nb_train_steps)
    #model.fit(x_train,y_train, verbose = 1,validation_data=(x_validate, y_validate), shuffle=False)
    return model
def predict_weather(model):
    current_data = functionsWrappup.getLastWeatherReportForLocation({'lat':46.77, 'lng':23.59})[:8]
    print(current_data)
    prediction_data = []
    date_of_tommorow = datetime.datetime.now() + datetime.timedelta(days=1)
    for el in current_data:
        row = []
        # row.append(float(date_of_tommorow.timestamp()))
        # row.append(float(el['lat']))
        # row.append(float(el['lng']))
        row.append(float(el['temperature_2m']))
        row.append(float(el['relative_humidity_2m']))
        row.append(float(el['precipitation']))
        row.append(float(el['rain']))
        row.append(float(el['showers']))
        row.append(float(el['snowfall']))
        row.append(float(el['weather_code']))
        row.append(float(el['surface_pressure']))
        prediction_data.append(row)
    print("data used for prediction")
    print(prediction_data)
    prediction_data = np.array(prediction_data)
    print("predicting smr ")
    prediction_data = prediction_data.reshape((prediction_data.shape[0], 1, 8))
    prediction = model.predict(prediction_data, verbose=1)
    print("predicted weather")
    for el in prediction:
        print("predicted weather : ==========================")
        for a in el:
            print(round(a,5))
def save_model(model):
    with open('model_v2.pkl', 'wb') as file:
        pickle.dump(model, file)

def load_model():
    with open('model_v2.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def get_weather_data():
    client = MongoClient("mongodb://localhost:27017")
    db = client.licentatest
    weather = []
    for el in db.weather.find():
        weather.append(el)
    client.close()
    print("colected a number of lines " + str(len(weather)))
    return weather

def validate_model(model,validation_data=[]):
    data = get_weather_data()[:1024]
    prepared_data = prepare_data(data)
    prepared_data.sort(reverse=True,key=lambda x: x[0])
    x_validate = []
    y_validate = []

    for el in prepared_data:
        x_validate.append(el)
        y_validate.append(el)
    x_validate = np.array(x_validate)
    y_validate = np.array(y_validate)
    x_validate = x_validate.reshape((x_validate.shape[0], 1, 8))
    y_plot = []
    x_plot = []
    for el in y_validate:
        print(el[0])
        y_plot.append(el[0])
    prediction = model.predict(x_validate, verbose=1)
    for el in prediction:
        print(el[0])
        x_plot.append(el[0])
    matplot.plot(y_plot,x_plot)
    matplot.show()


def main():
    print("Starting...")
    print("Getting weather data")
    data = get_weather_data()
    print("Data loaded, nr of documents " + str((data)))
    print("Preprocessing data...")
    prepared_data = prepare_data(data)
    print("Data succesfuly preprocessed")
    print("Setting up model")
    model = setup_model()
    print("Model ready")
    print("Training model")
    model = train(prepared_data,model)
    print("Model trained")
    print("Saving model")
    save_model(model)
    print("Model saved")
    print("Finishing...")
    print("andra")

def test(ok):
    if ok ==1:
        model = setup_model(8,8)
        data = get_weather_data()
        cox = prepare_data(data)
        model = train(cox,model)
        save_model(model)
    else :
        model = load_model()
        predict_weather(model)
        validate_model(model)
test(0)