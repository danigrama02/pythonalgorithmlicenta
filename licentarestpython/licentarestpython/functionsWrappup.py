import datetime
import math

import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from pymongo import MongoClient, GEOSPHERE
def getWeatherReportForLocationsAndDate(locations,date):
    pass

def getWeatherAlertsForLocationsAndDate(locations,date):
    pass

def predictWeather():
    pass

def getDataFromAroundthePoint(location):
    lat = float(location['lat'])
    lng = float(location['lng'])

    client = MongoClient("mongodb://localhost:27017")
    db = client.licentatest

    client.close()


def getLastWeatherReportForLocation(location):
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    current_date = datetime.date.today()-datetime.timedelta(days=1)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": location['lat'],
        "longitude": location['lng'],
        "hourly": ["temperature_2m", "cloud_cover", "precipitation",
                   "weather_code"],
        "timezone": "GMT",
        "start_date": str(current_date),
        "end_date": str(current_date)
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
    hourly_cloud_cover = hourly.Variables(1).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(2).ValuesAsNumpy()
    hourly_weather_code = hourly.Variables(3).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )}

    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["cloud_cover"] = hourly_cloud_cover
    hourly_data["precipitation"] = hourly_precipitation
    hourly_data["weather_code"] = hourly_weather_code

    hourly_dataframe = pd.DataFrame(data=hourly_data)
    json_hourly_dataframe = hourly_dataframe.to_dict(orient="records")

    return json_hourly_dataframe

#getLastWeatherReportForLocation({'lat':'46.77','lng':'23.59'})
#getDataFromAroundthePoint({'lat':'22','lng':'44'})
