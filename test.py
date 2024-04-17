import requests

print("start")
print("hello world")

api_key = "80a47ba913f346498ec93239242202"

url = "https://archive-api.open-meteo.com/v1/archive?latitude=52.52&longitude=13.41&start_date=2010-01-01&end_date=2010-02-02&hourly=temperature_2m,relative_humidity_2m,precipitation,rain,snowfall,snow_depth,weather_code,pressure_msl,surface_pressure,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,wind_speed_10m,wind_speed_100m,wind_direction_10m,wind_direction_100m,soil_temperature_0_to_7cm,soil_moisture_0_to_7cm,is_day,sunshine_duration&models=best_match"

#response = requests.get(url)

#response_json = response.json()

#f = open("data.txt","w")
#for el in response_json:
#    print(el)
#    print(response_json[el])
#f.write(str(response_json))
#f.close()

print("end")


import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry
from pymongo import MongoClient

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://api.open-meteo.com/v1/forecast"
params = {
	"latitude": 52.52,
	"longitude": 13.41,
	"hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "rain", "showers", "snowfall", "weather_code", "surface_pressure"],
	"timezone": "GMT",
	"start_date": "2020-04-05",
	"end_date": "2024-04-10"
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
hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
hourly_precipitation = hourly.Variables(2).ValuesAsNumpy()
hourly_rain = hourly.Variables(3).ValuesAsNumpy()
hourly_showers = hourly.Variables(4).ValuesAsNumpy()
hourly_snowfall = hourly.Variables(5).ValuesAsNumpy()
hourly_weather_code = hourly.Variables(6).ValuesAsNumpy()
hourly_surface_pressure = hourly.Variables(7).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}
hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
hourly_data["precipitation"] = hourly_precipitation
hourly_data["rain"] = hourly_rain
hourly_data["showers"] = hourly_showers
hourly_data["snowfall"] = hourly_snowfall
hourly_data["weather_code"] = hourly_weather_code
hourly_data["surface_pressure"] = hourly_surface_pressure

hourly_dataframe = pd.DataFrame(data = hourly_data)
json_hourly_dataframe = hourly_dataframe.to_dict(orient="records")
#f = open("data.json","a")
#f.write("/n"+str(hourly_dataframe.to_json(orient="table")))
#f.close()

print(type(json_hourly_dataframe))

client = MongoClient("mongodb://localhost:27017")
db = client.licentatest
try: db.command("serverStatus")
except Exception as e: print(e)
else: print("You are connected!")

for el in json_hourly_dataframe:
	db.weather.insert_one(el)

#import requests
#api_url = "http://api.weatherapi.com/v1/history.json?key=80a47ba913f346498ec93239242202&q=bulk"
#response = requests.get(api_url)
#print(response.json())
#
for el in db.weather.find():
    print(el)
client.close()
