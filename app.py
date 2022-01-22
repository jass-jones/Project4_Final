import numpy as np
from flask import Flask, request, jsonify, render_template
# from logging import debug
# from flask_pymongo import PyMongo
# from werkzeug.utils import append_slash_redirect, redirect
import pickle
# import easygui

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# mongo = PyMongo(app, uri="mongodb://localhost:27017/city_weather_db")

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]
    prediction = model.predict(final_features)
    if prediction == 1:
        return render_template('index.html', prediction_text = f"This song is predicted to make the Spotify Top 200 Charts!")
    else:
        return render_template('index.html', prediction_text = f"This song is not predicted to make the Spotify Top 200 Charts.")


if __name__ == '__main__':
    app.run(debug=True)