# File Description : Flask Server. Communicate between frontend (phone) and backend (AI stuff)

# Does not finish (keeps running forever)

import os

# import pose_processing

from flask import Flask, request, jsonify


# initializing our server
app = Flask(__name__)


# @app.route("/")
# def welcome():
#     return "Welcome to ShengLin's Server!"


@app.route("/analyze")
def analyze_videos():
    return "Analyze"


# run the server locally
app.run(host='0.0.0.0')