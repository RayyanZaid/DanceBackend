import os
from pathlib import Path
#machine learning model Imports
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import ParameterGrid

import math

import db

#Main function
def process_videos(videoPath1 , videoPath2):
    print ("Processing Videos")




 #step 1 to mp4
    videoPath1= convertToMp4(videoPath1)
    videoPath2= convertToMp4(videoPath2)
    #we want to get frame angles
    #frame angles-- array
 
 
 #getting date 
    video1FrameData , imageName1 = get_frames_angles(videoPath1)
    video2FrameData , imageName2 = get_frames_angles(videoPath2)


    #3) using machine learning
    video1KeyFrames =[]
    video2KeyFrames=[]
    public_urls = get_image_urls(video1FrameData,video2FrameData,imageName1,imageName2,video1KeyFrames,video2KeyFrames)

    averageError = calculate_average_error(video1KeyFrames ,video2KeyFrames)
    suggestion = analyze_dance_quality(averageError)

    return  averageError,public_urls,suggestion



#3) using machine learning


# finds the distance btwn 2 images and returns it
def euclidean_distance(p1, p2):
    if len(p1) != len(p2):
        raise ValueError("Lists must have the same number of elements")
    
    squared_diff_sum = sum((x - y) ** 2 for x, y in zip(p1, p2))
    distance = math.sqrt(squared_diff_sum)
    return distance


from datetime import datetime

def get_image_urls(studentVideoFrameData,professionalVideoFrameData,studentFolderName,professionalFolderName,studentVideoKeyFrames,professionalVideoKeyFrames):
    

    #1)use the student data (video1FrameData) and create the clusters

    student_cluster = get_cluster(studentVideoFrameData)
    #2) creat 2 groups of cluster and compare

    public_urls = []

    imgNum = 1

    # Get the current date and time
    now = datetime.now()

    # Format the date and time as a string
    current_date_time = now.strftime("%Y-%m-%d %H:%M:%S")


    for label in student_cluster:
        
        if label['start'] == None or label['end'] == None:
            continue
        
        
        index_student = (label['start']+label['end'])//2

        student_image_file = f"{studentFolderName}/{index_student}.jpg"


        studentFrame = studentVideoFrameData[index_student]   # [1,2,3,4,5,6,7,8] - angles of the body parts



        # 10 seconds at 30 FPS
        # professionalVideoFrameData = [
        #   [1,2,3,4,5,6,7,8],      # Frame 0
        #   [1,2,3,4,5,6,7,8],      # Frame 1
        #   [1,2,3,4,5,6,7,8],      # Frame 299
        # ]

        smallestDistance = float('inf')
        index_professional = 0 

        i = 0

        for eachProFrame in professionalVideoFrameData:
            
            currDistance = euclidean_distance(studentFrame,eachProFrame)

            if currDistance < smallestDistance:
                index_professional = i
                smallestDistance = currDistance

            i = i + 1
        
        professional_image_file = f"{professionalFolderName}/{index_professional}.jpg"


        #TODO: add the 2 more params
        urls = db.send_data(student_image_file,professional_image_file, imgNum, current_date_time)

        imgNum += 1
        public_urls.append(urls)






        #TODO: Send student and professional images to database

        #update key frames to cpmpare

        studentVideoKeyFrames.append(studentVideoFrameData[index_student])
        professionalVideoKeyFrames.append(professionalVideoFrameData[index_professional])

        

            

    return public_urls
    




def get_cluster(video1FrameData):
    numCluster = kmean_hyper_param_tuning(video1FrameData)

    if numCluster > 10:
        numCluster = 10
    X = np.array(video1FrameData)
    # creat Kmeans model (best line of fit)with 'n' clusters using our video1data
    kmean_1 = KMeans(n_clusters=numCluster).fit(X)

    student_cluster = []

  #In the get_cluster function, do the same thing as the shopping list for loop I showed in class
    start = 0 
    label = None
    end =None

    labels= kmean_1.labels_
    
    for i in range(1, len(labels)):

        if len(student_cluster) >= numCluster:
            break

        
        if labels [i] != labels[i-1]:

            end = i - 1
            student_cluster.append(
                {
                    'start':start,
                    'end': i - 1,
                    'label': labels[i-1]



                }
            )

            start = i


        else:

            # last cluster

            end = i
            student_cluster.append(

                {
                    'start':start,
                        'end': i,
                        'label': labels[i]
                }
            )

            start = i + 1

    return student_cluster




# the number of key frames
# n_clusters -- how many stages (picture)there will be

#purpose : determine how many pictures

def kmean_hyper_param_tuning(video1FrameData):

    parameters = []

    for i in range(2, 31):
        parameters.append(i)


    parameter_grid = ParameterGrid(
        {'n_clusters':parameters}
    )
    #go through params in parameter_grid

    best_score = -1
    best_grid = {}

    kmeans_model = KMeans()

    for p in parameter_grid:
        kmeans_model.set_params(**p)
        kmeans_model.fit(video1FrameData)

        ss = metrics.silhouette_score(video1FrameData,kmeans_model.labels_)

        print("Parameter:",p,'Score:',ss)
        
        if ss > best_score:
            best_score = ss
            best_grid['n_clusters'] = p

    return best_grid['n_clusters']['n_clusters']

def calculate_average_error(studentFrames , professionalFrames):
    averageError = 0
   
    studentFrames = np.array(studentFrames)
    professionalFrames = np.array(professionalFrames)

    differences = abs(professionalFrames - studentFrames)


    allDifference = []

    for array in differences:
        
        for  dif in range(8):
            
            if array[dif] >= 15 :
                allDifference.append(array[dif])
    errorTotal = len(allDifference)

    averageError = (  errorTotal / (len(differences)*8)   ) *100


    print("Your score" + str(averageError))

    return averageError


   
 

# Average error - anywhere from 0 -- 100
def analyze_dance_quality(average_error):
    # if average_error is between 0 -- 5 (including 0 and 5)
    # 0 -- 5

    if average_error >=0 and average_error<= 5:
        return  "Outstanding! Your dance performance is exceptional. Consider experimenting with complex choreography and unique movements to further elevate your skills. "

   

    # 6 -- 10
    if   average_error >=6 and average_error<= 10:
        return "Excellent Dance Performance. Your technique is nearly flawless. Try incorporating more expression  \
            and emotion into your movements for an even more captivating performance. "
    # 11 -- 15
    if   average_error >=11 and average_error<= 15:
        return"Very Impressive! Your dance quality is excellent with only minor imperfections. Focus on refining  \
               transitions and adding your personal touch to make your performance truly memorable. "
    # 16 -- 20
    if   average_error >=16 and average_error<= 20:
        return"Great Job! Your dance performance is strong. Work on perfecting specific poses and movements to  \
                enhance overall fluidity and grace. "

    # 21 -- 30
    if   average_error >=21 and average_error<= 30:
        return"Good Dance Performance. You're doing well, but there's room for improvement. Pay attention to details  \
                and explore variations in your dance routine to keep it engaging. "

    # 31 -- 40
    if   average_error >=31 and average_error<= 40:
        return "Competent Dance Performance. Your dance quality is solid, but there are noticeable areas for  \
                improvement. Practice specific movements and experiment with different styles to broaden your \
        repertoire. "

    if  average_error >=41 and average_error<= 50:
        return"Fair Dance Performance. Your dance skills are average. Focus on mastering fundamental techniques,  \
               improving coordination, and maintaining good posture throughout your routine. "
    
    if  average_error >=51 and average_error<= 60:
        return"Needs Improvement. Significant improvement is required in various aspects of your dance performance.  \
                Consider seeking guidance from a dance instructor and dedicating more time to practice. "
    # ... ... 
    if  average_error >=61 and average_error<= 70:
        return "Below Average. Your dance quality needs substantial improvement. Work on foundational movements,  \
             posture, and timing. Regular practice and feedback from an instructor can make a significant  \
                difference. "
    if  average_error >=71 and average_error<= 80:
        return"Poor Dance Quality. Your performance is below expectations. Revisit basic dance principles, \
               refine coordination, and seek personalized coaching to address specific weaknesses. "
    if  average_error >=81 and average_error<= 90:
        return "Very Low Dance Quality. Your dance skills are significantly below the desired standard. Consider starting with the basics, focusing on rhythm, and seeking intensive training to build a strong foundation. "
    # 91 -- 100
    if  average_error >=91 and average_error<= 100:
        return "Extremely Low Dance Quality. Substantial improvement is needed in every aspect of your dance performance. Consider enrolling in beginner dance classes to develop fundamental skills and  techniques. "






    

# getting data
def make_directory(name:str):
    # if the directory does not exist ,create it
    if  not os.path.isdir(name):
        os.mkdir(name)







from moviepy.editor import VideoFileClip

def convertToMp4(path):
    _,extension = os.path.splitext(path)
    if extension != ".mp4":
        
        
        
        mp4_path = Path(path).stem + ".mp4"
        clip = VideoFileClip(path)
        clip.write_videofile(mp4_path,codec='libx264')
        
        
       
        return mp4_path
    return path





import cv2
def get_frames_angles(video_path)->tuple:
    
    frame_angles:list =[]
    basename = os.path.basename(video_path)
    image_name ,_= os.path.splitext(basename)
    
    make_directory(image_name)
    
    pose,poseDrawing = initializePoseTools()

    
    
    # frames = [
        
    #     [45.6, 78.1, 98.1],     # 1
    #      [5.6, 34.1, 98.1],     # 2
    #       [9.6, 78.1, 98.1],   
    #        [45.6, 78.1, 98.1], 
                        
    # ]
   
   
   
    cap = cv2.VideoCapture(video_path)


    img_count=0


    with pose.Pose(
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5
    )as poseModel:
        while cap.isOpened():

            success, frame =cap.read()
            if not success:

                print ("Could not read the frame")
                break        
            # If the program gets to here


            # 1) Pose Process the Frame 
                # Annotate the image with the lines and circles on body parts
                # Give us the Landmark Results -- we need this to get the angles
            
            #put the pose model
            poes_process_frame , landmark_result = pose_process_image(frame,poseModel)
            
            
            
            if landmark_result.pose_landmarks != None:
                #test
                image = draw_landmarks(landmark_result, poseDrawing, pose ,poes_process_frame)



                # cv2.imshow("Frame", image)
                # cv2.waitKey(1)

                # 2) Get the Angles
                h,w, _ = image.shape
                angles = get_angles_for_each_frame(pose, landmark_result, image,h ,w )


                # 3) Save the frame and save the angles 
                frame_angles.append(angles)

                imageFilePath = f"{image_name}/{img_count}.jpg"
                img_count += 1 
                cv2.imshow('Video',image)
                cv2.waitKey(1)
                cv2.imwrite(imageFilePath,image)





    return (frame_angles,image_name)
   
import mediapipe as mp
from mediapipe.python.solutions.pose import Pose

import numpy as np

def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    #use arc to calculate the angle
    radians =np.arctan2(c[1]-b[1],c[0]-b[0])   -   np.arctan2(a[1] - b[1] , a[0] - b[0])

    degrees = np.abs(radians* 180.0/np.pi)
    if degrees > 180.0:
        degrees = 360 - degrees
   
    return round(degrees,1)
    


def draw_angle(actualCoordinate , image ,angle):

    
    angleStr= str(angle)
    font= cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.4
    color = (255,255,255)
    thickness = 1 

    drawImage = cv2.putText(image,angleStr,actualCoordinate,font,fontScale,color,thickness)
    return drawImage




def plot_angle(p1,p2,p3,landmark_result,image,h ,w):
   
   landmark_result = landmark_result.pose_landmarks.landmark
   a = [    landmark_result[p1].x  ,  landmark_result[p1].y   ]
   b = [    landmark_result[p2].x ,  landmark_result[p2].y   ]
   c = [    landmark_result[p3].x ,   landmark_result[p2].y  ]
   angle = calculate_angle(a,b,c)

   actualXCoordiante = int(b[0]*w)
   actualYCoordinate = int(b[1]*h)
   actualCoordinate = (actualXCoordiante,actualYCoordinate)
   drawnImage=draw_angle(actualCoordinate, image, angle)

   return angle, drawnImage



    
def get_angles_for_each_frame(mp_pose, landmarks, image,h ,w):
   # 6 angles           
    angles = [ ]
    val = 50

    angle, image = plot_angle(mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                              mp_pose.PoseLandmark.LEFT_ELBOW.value,
                              mp_pose.PoseLandmark.LEFT_WRIST.value, landmarks, image, h, w + val)
    angles.append(angle)
    angle, image = plot_angle(mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                              mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                              mp_pose.PoseLandmark.RIGHT_WRIST.value, landmarks, image, h, w - val)
    angles.append(angle)
    angle, image = plot_angle(mp_pose.PoseLandmark.LEFT_HIP.value,
                              mp_pose.PoseLandmark.LEFT_KNEE.value,
                              mp_pose.PoseLandmark.LEFT_ANKLE.value, landmarks, image, h, w + val)
    angles.append(angle)
    angle, image = plot_angle(mp_pose.PoseLandmark.RIGHT_HIP.value,
                              mp_pose.PoseLandmark.RIGHT_KNEE.value,
                              mp_pose.PoseLandmark.RIGHT_ANKLE.value, landmarks, image, h, w - val)
    angles.append(angle)

    angle, image = plot_angle(mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                              mp_pose.PoseLandmark.LEFT_HIP.value,
                              mp_pose.PoseLandmark.LEFT_KNEE.value, landmarks, image, h, w + val)
    angles.append(angle)
    angle, image = plot_angle(mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                              mp_pose.PoseLandmark.RIGHT_HIP.value,
                              mp_pose.PoseLandmark.RIGHT_KNEE.value, landmarks, image, h, w - val)
    angles.append(angle)

    angle, image = plot_angle(mp_pose.PoseLandmark.LEFT_WRIST.value,
                             mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                             mp_pose.PoseLandmark.LEFT_HIP.value, landmarks, image, h, w + val)
    angles.append(angle)
    angle_wrist_shoulder_hip_right, image = plot_angle(mp_pose.PoseLandmark.RIGHT_WRIST.value,
                                                       mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                                                       mp_pose.PoseLandmark.RIGHT_HIP.value, landmarks, image, h,
                                                       w - val)
    angles.append(angle_wrist_shoulder_hip_right)

    # cv2.imshow('Hopefully this works' , image)
    # cv2.waitKey(1)

    return angles


def draw_landmarks(results, mp_drawing, mp_pose, image):
    # for idx (index), x (value) in enumerate(_____):   \\storing both the index and the value
    # work w/both variables simultaneously; requires

    if results == None:
        print("Cannot  see all the angles")
    elif results.pose_landmarks == None:
        print("No landmarks")
    else:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            # we care about 11-16 and 23-28
            if idx in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22, 29, 30, 31, 32]:
                results.pose_landmarks.landmark[idx].visibility = 0  # remove visibility of specific landmarks

        # draw landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                                # customize color, etc
                                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

    return image


def initializePoseTools():

    pose = mp.solutions.pose

    mp_drawing = mp.solutions.drawing_utils


    return pose , mp_drawing

def pose_process_image(openCVFrame,poseModel : Pose):
    
    rgbImage =  cv2.cvtColor(openCVFrame,cv2.COLOR_BGR2RGB)
    
    
    landmark_results = poseModel.process(rgbImage)
    # if landmark_results.pose_landmarks:

    #     for id, landmark in enumerate(landmark_results.pose_landmarks.landmark):

    #         print(id)
    #         print(f"x: {landmark.x}")
    #         print(f"y: {landmark.y}")
    #         print(f"z: {landmark.z}")

    #         print()
    
    
    
    openCVFrame = cv2.cvtColor(rgbImage , cv2.COLOR_RGB2BGR)

   

    return( openCVFrame ,landmark_results)



if __name__ == '__main__':
    process_videos("Student Dance.mp4","Tutorial Dance.mp4")