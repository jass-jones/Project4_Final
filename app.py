from flask import Flask, render_template
from logging import debug
from flask_pymongo import PyMongo
from werkzeug.utils import append_slash_redirect, redirect
from script import api_call
import easygui

app = Flask(__name__)

mongo = PyMongo(app, uri="mongodb://localhost:27017/city_weather_db")

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/script")
def script():
    api_call()
    easygui.msgbox("Database Updated Successfully\nCheck Your Mongo To See Our Raw Data", "Success")
    return redirect("/")


if __name__ == '__main__':
    app.run(debug=True)