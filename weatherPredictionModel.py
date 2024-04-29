import datetime
import matplotlib.pyplot as matplot

from pymongo import MongoClient
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
import numpy as np
import pickle
import random
import functionsWrappup
from sklearn.metrics import mean_squared_error, r2_score
'''
Function used to set up the model.
:return: model
'''
def setup_model():
    model = Ridge(alpha=0.1)
    return model

'''
Function used to convert the recived data from the db into training data.
:param data: list of documents from the db containing the weather data
:return: a converted list with entries representing the weather data as list
'''
def prepare_data(data):
    prepared_data = []
    for el in data:
        row = []
        row.append(float(el['date'].timestamp()))
        # row.append(float(el['lat']))
        # row.append(float(el['lng']))
        row.append(float(el['temperature_2m']))
        row.append(float(el['cloud_cover']))
        row.append(float(el['precipitation']))
        #row.append(float(el['rain']))
        #row.append(float(el['showers']))
        #row.append(float(el['snowfall']))
        row.append(float(el['weather_code']))
        #row.append(float(el['surface_pressure']))
        prepared_data.append(row)
    return prepared_data
'''
Function used for the model training, 
the preprocesed data is split into training samples and validations semples and than feed the model
also through the sample_size param it will be training with only that many samples
:param data: list of preprocessed data
:param sample_size: the number of samples used for training
:return: trained model
'''
def train(data, sample_size = 1024):
    model = Ridge(alpha=0.1,positive=True, solver='lbfgs')
    x_train = []
    y_train = []
    x_validate = []
    y_validate = []
    i = 0
    n = len(data)
    while i<n-1:
        if random.random() < 0.8:
            #x_train.append(data[i][:1])
            #y_train.append(data[i+1][1:])
            x_train.append(data[i])
            y_train.append(data[i+1][1:])
        else:
            x_validate.append(data[i])
            y_validate.append(data[i+1][1:])
            #x_validate.append(data[i][:1])
            #y_validate.append(data[i+1][1:])
        i+=2

    print("fiitinf model")
    print("====================================")
    model.fit(x_train, y_train)
    predictions = model.predict(x_validate)
    mse = mean_squared_error(y_validate, predictions)
    r2 = r2_score(y_validate, predictions)
    print(f" MSE = {mse}, R^2 = {r2}")

    print("done")

    return model
'''
Function used to predict the weather for the next day
:param model: the trained model
:return : list containing the predicted weather
'''
def predict_weather(model):
    current_data = functionsWrappup.getLastWeatherReportForLocation({'lat':46.77, 'lng':23.59})
    print(current_data)
    print("seklected hours nr " + str(len(current_data)))
    prediction_data = []
    date_of_tommorow = datetime.datetime.now() + datetime.timedelta(days=1)
    for el in current_data:
        row = []
        row.append(float(date_of_tommorow.timestamp()))
        # row.append(float(el['lat']))
        # row.append(float(el['lng']))
        row.append(float(el['temperature_2m']))
        row.append(float(el['cloud_cover']))
        row.append(float(el['precipitation']))
        # row.append(float(el['rain']))
        # row.append(float(el['showers']))
        # row.append(float(el['snowfall']))
        row.append(float(el['weather_code']))
        # row.append(float(el['surface_pressure']))
        prediction_data.append(row)
    print("data used for prediction")
    print(prediction_data)
    prediction_data = np.array(prediction_data)
    print("predicting smr ")
    #prediction_data = polynomial_features.fit_transform(prediction_data)
    for el in prediction_data[0]:
        print("_----------------------")
        print(el)
    prediction = model.predict(prediction_data)
    print("predicted weather")
    for el in prediction:
        print("predicted weather : ==========================")
        for a in el:
            print(round(a,5))
    temp = []
    time = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
    for el in prediction:
        temp.append(el[0])
    matplot.plot(time, temp, label='Predicted temperature')
    matplot.legend()
    matplot.show()
'''
Function used to save the model as a pkl file.
:param model: the model to be saved
:param model_name: the name of the file
'''
def save_model(model,model_name):
    with open(model_name, 'wb') as file:
        pickle.dump(model, file)
'''
Function used to load the model from the pkl file.
:param model_name: the name of the file
:return: the loaded model
'''
def load_model(model_name):
    with open(model_name, 'rb') as file:
        model = pickle.load(file)
    return model

'''
Function used to get the weather data from the db
:return: list of documents containing the weather data
'''
def get_weather_data():
    client = MongoClient("mongodb://localhost:27017")
    db = client.licentatest
    weather = []
    i = 0
    for el in db.weather_validation.find():
        if 'cloud_cover' in el.keys():
            weather.append(el)
        # if i == 30000:
        #     break
        # i+=1
    client.close()
    print("colected a number of lines " + str(len(weather)))
    return weather

'''
Function used to validate the model by plotting the predicted weather against the actual weather
:param model: the trained model
:param validation_data: list of documents containing the weather data used for validation
'''
def validate_model(model,data):
    time = []
    for el in data[:len(data)//2]:
        time.append(float(el['date'].timestamp()))

    prepared_data = prepare_data(data)
    x_validate = []
    y_validate = []
    i = 0
    n = len(prepared_data)
    while i < n-1:
        x_validate.append(prepared_data[i])
        y_validate.append(prepared_data[i+1][1:])
        i+=2
    # x_validate = np.array(x_validate)
    # y_validate = np.array(y_validate)

    prediction = model.predict(x_validate)
    print("Mse score : " + str(mean_squared_error(y_validate, prediction)))
    print("R2 score : " + str(r2_score(y_validate, prediction)))
    predicted_temp = []
    actual_temp = []
    for el in y_validate :
        actual_temp.append(el[0])
    for el in prediction:
        predicted_temp.append(el[0])
    prediction2 = []
    for el in prediction:
        row = []
        for a in el:
            row.append(round(a,5))
        prediction2.append(row)
    for i in range(0,len(prediction2)):
        print(prediction2[i])
        print(y_validate[i])
    matplot.plot(time, actual_temp, label='Actual temperature')
    matplot.plot(time, predicted_temp, label='Predicted temperature')
    matplot.legend()
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
    model_name = 'model_v13.pkl'
    if ok ==1:
        #model = setup_model(8,8)
        data = get_weather_data()
        cox = prepare_data(data)
        model = train(cox,)
        save_model(model,model_name)
    else :
        model = load_model(model_name)
        #predict_weather(model)
        data = get_weather_data()[:1024]
        validate_model(model,data)
#test(1)
test(0)