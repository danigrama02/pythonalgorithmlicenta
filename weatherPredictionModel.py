import datetime

from pymongo import MongoClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np
import pickle

def train():
    print("Starting getting data")
    client = MongoClient("mongodb://localhost:27017")
    db = client.licentatest

    try:
        db.command("serverStatus")
    except Exception as e:
        print(e)
    else:
        print("You are connected!")

    weather_data = []
    for el in db.weather.find():
        weather_data.append(el)

    print("finishing job")
    print("closing client")
    client.close()
    print("lenght of data geddered")
    print(len(weather_data))

    train_data = weather_data[0:35000]
    validation_data = weather_data[35000:35352]

    print("lenght of data fortrain and validation")
    print(len(train_data))
    print(len(validation_data))

    print("Starting ai")

    X_train = []
    y_train = []
    for el in train_data:
        X_train.append([float(el['temperature_2m'])])
        y_train.append([float(el['temperature_2m'])])

    # Convert X_train and y_train to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Get indices where y_train is not NaN
    not_nan_indices = np.where(~np.isnan(y_train))

    # Use these indices to filter X_train and y_train
    X_train = X_train[not_nan_indices]
    y_train = y_train[not_nan_indices]
    X_train = X_train.reshape(-1, 1)
    X_validation = []
    y_validation = []

    for el in validation_data:
        X_validation.append([float(el['temperature_2m']),str(el['date']),float(el['relative_humidity_2m']),float(el['precipitation']),float(el['rain']),float(el['showers']),float(el['snowfall']),float(el['weather_code']),float(el['surface_pressure'])])
        y_validation.append([float(el['temperature_2m']),str(el['date']),float(el['relative_humidity_2m']),float(el['precipitation']),float(el['rain']),float(el['showers']),float(el['snowfall']),float(el['weather_code']),float(el['surface_pressure'])])


    print("Starting training")
    model = RandomForestRegressor(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)

    print("Starting validation")
    y_pred = model.predict(X_validation)
    print("Mean squared error: ", mean_squared_error(y_validation, y_pred))
    print("Cross validation score: ", np.mean(cross_val_score(model, X_train, y_train, cv=5)))

    print("Saving model")

    with open("weather_prediction_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved")

    print("End of the program")


def predict_weather(model):
    return model.predict()

def main():
    train()
    with open("weather_prediction_model.pkl", "rb") as f:
        model = pickle.load(f)
    print(predict_weather(model))
main()