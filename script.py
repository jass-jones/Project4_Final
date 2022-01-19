import pandas as pd
import numpy as np
import requests
import time
from config import api_key
import pymongo

def api_call():
    #connect to mongo db
    client = pymongo.MongoClient("mongodb://localhost:27017")

    #getting city list of top 200 cities

    cities = pd.read_csv(r"static\resources\cities.csv")
    cities.head()

    city_list = []

    for city in cities['name']:
        city_list.append(city)

    #first api call for current weather data

    url = 'http://api.openweathermap.org/data/2.5/weather?'
    units = 'metric'
    query_url = f'{url}appid={api_key}&units={units}&q='

    city_name = []
    lat = []
    lng = []
    max_temp = []
    humidity = []
    cloudiness = []
    wind_speed = []
    country = []
    date = []
    for city in city_list:
        response = requests.get(query_url + city).json()
        city_name.append(city)
        lat.append(response['coord']['lat'])
        lng.append(response['coord']['lon'])
        max_temp.append(response['main']['temp_max'])
        humidity.append(response['main']['humidity'])
        wind_speed.append(response['wind']['speed'])
        country.append(response['sys']['country'])
        cloudiness.append(response['clouds']['all'])
        date.append(response['dt'])

    #creating dataframe and saving to new csv when called

    weather = {'City':city_name,'Lat':lat,'Lng':lng,'Max Temp':max_temp,'Humidity':humidity,'Cloudiness':cloudiness,'Wind Speed':wind_speed,'Country':country,'Date':date}
    weather_df = pd.DataFrame({ key:pd.Series(value) for key, value in weather.items() })
    weather_df.to_csv(r'static\resources\weather_df.csv', index=False)

    #update mongo db with new data

    weather_data = weather_df.to_dict(orient='records')
    db = client["city_weather_db"]
    db.weather.insert_many(weather_data)

    #second api call for 5 day forecast

    url = 'http://api.openweathermap.org/data/2.5/forecast?'
    units = 'metric'
    query_url = f'{url}appid={api_key}&units={units}&q='


    city_name = []
    lat_5 = []
    lng_5 = []
    max_temp_5 = []
    humidity_5 = []
    cloudiness_5 = []
    wind_speed_5 = []
    date_5 = []


    for city in city_list:
        for i in range(0,40,8):
                response = requests.get(query_url + city).json()
                city_name.append(city)
                lat_5.append(response['city']['coord']['lat'])
                lng_5.append(response['city']['coord']['lon'])
                max_temp_5.append(response['list'][i]['main']['temp_max'])
                humidity_5.append(response['list'][i]['main']['humidity'])
                wind_speed_5.append(response['list'][i]['wind']['speed'])
                cloudiness_5.append(response['list'][i]['clouds']['all'])
                date_5.append(response['list'][i]['dt_txt'])

    #creating dataframe and saving to new csv when called
    weather_5 = {'City':city_name,'Lat':lat_5,'Lng':lng_5,'Max Temp':max_temp_5,'Humidity':humidity_5,'Cloudiness':cloudiness_5,'Wind Speed':wind_speed_5,'Date':date_5}
    weather5_df = pd.DataFrame({ key:pd.Series(value) for key, value in weather_5.items() })
    weather5_df.to_csv(r'static\resources\weather_5.csv', index=False)

    #update mongo db with new data
    data5 = weather5_df.to_dict(orient='records')
    db = client["city_weather_db"]
    db.weather_5.insert_many(data5)

