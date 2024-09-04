import os
import shutil
from flask import Flask, request, jsonify
from pose_processing import process_videos

# initializing our server
app = Flask(__name__)

@app.route("/analyze", methods=['POST'])
def analyze_videos():

    if 'studentVideo' not in request.files or 'professionalVideo' not in request.files:
        return jsonify({"error": "Both studentVideo and professionalVideo are required."}), 400
    
    # Get the data from the phone. Store it in variables
    student_files = request.files.getlist("studentVideo")
    professional_files = request.files.getlist("professionalVideo")
    
    # Check if files are uploaded correctly
    if len(student_files) == 0 or len(professional_files) == 0:
        return jsonify({"error": "Both studentVideo and professionalVideo are required."}), 400

    # Assuming only one file per field is expected; take the first file
    studentVideo = student_files[0] if len(student_files) > 0 else None
    professionalVideo = professional_files[0] if len(professional_files) > 0 else None

    if not studentVideo or not professionalVideo:
        return jsonify({"error": "Both studentVideo and professionalVideo are required."}), 400

    studentVideoName = studentVideo.filename
    professionalVideoName = professionalVideo.filename

    # Save videos if they have valid names
    if len(studentVideoName) > 0 and len(professionalVideoName) > 0:
        print("Both are valid videos")
        studentVideo.save(studentVideoName)
        professionalVideo.save(professionalVideoName)
    else:
        return jsonify({"error": "Invalid video files."}), 400

    # Process videos
    averageError, public_urls, suggestions = process_videos(studentVideoName, professionalVideoName)

    # Clean up saved files
    os.remove(studentVideoName)
    os.remove(professionalVideoName)

    studentVideoNameWithoutExt , _ = os.path.splitext(studentVideoName)
    professionalVideoNameWithoutExt , _ = os.path.splitext(professionalVideoName)
    
    studentImageFolder = studentVideoNameWithoutExt
    professionalImageFolder = professionalVideoNameWithoutExt
    
    shutil.rmtree(studentImageFolder)
    shutil.rmtree(professionalImageFolder) 

    return jsonify({
        'averageError': averageError,
        'public_urls': public_urls,
        'suggestions': suggestions
    }), 200

# # Run the server locally
# if __name__ == "__main__":
#     app.run(host='127.0.0.1', port=8000)

# Run the server publically
if __name__ == "__main__":
    app.run(host="0.0.0.0")
