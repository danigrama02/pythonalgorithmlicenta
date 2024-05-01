import math

import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from pymongo import MongoClient

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)


# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below


def gettherWaether(lat, lng):
    print("start gathering for " + str(lat) + " " + str(lng))
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": float(lat)/10.0,
        "longitude": float(lng)/10.0,
        "start_date": "2024-01-01",
        "end_date": "2024-04-01",
        "hourly": ["temperature_2m", "precipitation", "weather_code", "cloud_cover"]
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()
    hourly_weather_code = hourly.Variables(2).ValuesAsNumpy()
    hourly_cloud_cover = hourly.Variables(3).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )}
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["precipitation"] = hourly_precipitation
    hourly_data["weather_code"] = hourly_weather_code
    hourly_data["cloud_cover"] = hourly_cloud_cover

    hourly_dataframe = pd.DataFrame(data=hourly_data)
    json_hourly_dataframe = hourly_dataframe.to_dict(orient="records")

    print(type(json_hourly_dataframe))

    client = MongoClient("mongodb://localhost:27017")
    db = client.licentatest
    try:
        db.command("serverStatus")
    except Exception as e:
        print(e)
    else:
        print("You are connected!")

    for el in json_hourly_dataframe:
        print(el)
        if not math.isnan(el['temperature_2m']):
            db.weather_v2.insert_one(el)
            print("Inserted")
            print(el)

    #for el in db.weather.find():
    #    print(el)
    client.close()

def main():
     for i in range(220, 280):
        for j in range(440, 480):
            gettherWaether(float(i), float(j))

#main()

# client = MongoClient("mongodb://localhost:27017")
# db = client.licentatest
# try:
#     db.command("serverStatus")
# except Exception as e:
#     print(e)
# else:
#     print("You are connected!")
#
# marked = []
# for el in db.weather_validation.find():
#     if 'cloud_cover' not in el.keys():
#         marked.append(el)
#         print(el)
# for el in marked:
#     db.weather_validation.delete_one(el)
# client.close()
#gettherWaether(223,465)
