import numpy as np
from flask import Flask, request, jsonify, render_template
# from logging import debug
# from flask_pymongo import PyMongo
# from werkzeug.utils import append_slash_redirect, redirect
import pickle
# import easygui

app = Flask(__name__, template_folder= 'Templates')
classifier = pickle.load(open('classifier.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# mongo = PyMongo(app, uri="mongodb://localhost:27017/city_weather_db")

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]
    prediction1 = classifier.predict(final_features)
    prediction2 = model.predict(final_features)
    if prediction1 == 1 and prediction2 == 0:
        return render_template('index.html', prediction1_text = f"This song is predicted to make the Spotify Top 200 Charts!", prediction2_text = f"It will likely be on the charts for 1-4 weeks.")
    elif prediction1 == 1 and prediction2 == 1:
        return render_template('index.html', prediction1_text = f"This song is predicted to make the Spotify Top 200 Charts!", prediction2_text = f"It will likely be on the charts for 5-24 weeks.")
    elif prediction1 == 1 and prediction2 == 2:
        return render_template('index.html', prediction1_text = f"This song is predicted to make the Spotify Top 200 Charts!", prediction2_text = f"It will likely be on the charts for 25-52 weeks.")
    elif prediction1 == 1 and prediction2 == 3:
        return render_template('index.html', prediction1_text = f"This song is predicted to make the Spotify Top 200 Charts!", prediction2_text = f"It will likely be on the charts for over a year!")
    else:
        return render_template('index.html', prediction1_text = f"This song is not predicted to make the Spotify Top 200 Charts.")

if __name__ == '__main__':
    app.run(debug=True)