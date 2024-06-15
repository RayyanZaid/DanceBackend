# File Description : Flask Server. Communicate between frontend (phone) and backend (AI stuff)

# Does not finish (keeps running forever)

import os

from flask import Flask, request, jsonify

from pose_processing import process_videos

# initializing our server
app = Flask(__name__)


# @app.route("/")
# def welcome():
#     return "Welcome to ShengLin's Server!"


@app.route("/analyze", methods = ['POST'])
def analyze_videos():
    
    # Get the data from the phone. Store it in a variable
    studentVideo = request.files.getlist("studentVideo")[0]
    professionalVideo = request.files.getlist("professionalVideo")[0]

    studentVideoName = studentVideo.filename
    professionalVideoName = professionalVideo.filename

    # if both filenames do not equal '' , then i want to save

    if len(studentVideoName) > 0 and len(professionalVideoName) > 0:
        print("Both are valid videos")
        studentVideo.save(studentVideoName)
        professionalVideo.save(professionalVideoName)
    

    averageError, public_urls, suggestions = process_videos(studentVideoName, professionalVideoName)

    os.remove(studentVideoName)
    os.remove(professionalVideoName)

    return jsonify({
        'averageError' : averageError,
        'public_urls' : public_urls,
        'suggestions' : suggestions
    }, 200)

    # HTTP Protocol - rules for how devices communicate over network
    # 200 - Successful
    
# run the server locally
app.run(host='0.0.0.0')