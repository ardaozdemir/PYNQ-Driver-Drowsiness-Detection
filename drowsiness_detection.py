#Import libraries
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
from pynq.overlays.base import BaseOverlay
from pynq.lib import Wifi
import json
import requests

#Import base overlays
base = BaseOverlay("base.bit")

#Connect to Wi-Fi
port = Wifi()
port.connect('Wi-Fi name', 'Wi-Fi password') #It needs to be edited

def distance(point_A, point_B):
    #Implementation of euclidean distance formula
    return np.linalg.norm(point_A - point_B)

def eye_aspect_ratio(eye):
    #Compute distances between 6 points
    A = distance(eye[2], eye[4])
    B = distance(eye[1], eye[5])
    C = distance(eye[0], eye[3])
    #Find EAR
    ear = (A + B) / (2.0 * C)
    #Return EAR
    return ear
 
 
 #Sound warning to the driver
 def sound_warning():
 
    pAudio = base.audio
    pAudio.load("/home/xilinx/jupyter_notebooks/base/audio/data/alarm.wav")
    pAudio.play()
    
    
 #Send information to cloud function
 def send_to_cloud(status):
    headers = {'Content-type': 'application/json'}

    url = 'https://iothook.com/api/update/'

    #Api key and driver status
    data = { 
    'api_key': 'd71499510d09073eea3675de', 
    'field_1': status,
    }

    #Sending driver status to cloud
    data_json = json.dumps(data)
    response = requests.post(url, data=data_json, headers=headers)
    
 
    
#Import haar cascade classifier
detector = cv2.CascadeClassifier(
    '/home/xilinx/jupyter_notebooks/base/video/data/'
    'haarcascade_frontalface_default.xml')

#Import facial landmark model
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#Determining thresholds
EAR_THRESHOLD = 0.22
EAR_COUNT = 5

COUNTER = 0

#Get indexes of left and right eye
(left_start, left_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_start, right_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


#Camera starts
vs = VideoStream(src=0).start()
time.sleep(1.0)

#Detection loop starts
while (True):
    #Get the frame from camera
    img = vs.read()
    #Resize the image
    img = imutils.resize(img, width=480)
    #Convert to black and white
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Detect faces using facial landmarks
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
    minNeighbors=4, minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE)
    
    #If there are no face found which means driver not looking straight
    if (len(rects) == 0):
        sound_warning()
        send_to_cloud("Drowsy!")
        continue
    
    #Get faces in rects array
    for (x, y, w, h) in rects:
    
        #Create rectangle for face to work with dlib
        face = dlib.rectangle(int(x), int(y), int(x + w),
            int(y + h))
            
        #Determine facial landmarks using the model
        shape = predictor(gray, face)
        #Convert coordinates to numpy array
        shape = face_utils.shape_to_np(shape)
        
        #Get the coordinates of both eyes
        left_eye = shape[left_start:left_end]
        right_eye = shape[right_start:right_end]
        
        #Calculate the EAR of both eyes
        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)
        #Take the average of both eyes
        EAR = (left_EAR + right_EAR) / 2.0

        #Check if it is lower than the threshold
        if EAR < EAR_THRESHOLD:
            COUNTER += 1
            #If eyes are closed for a long time warning the driver and send status to cloud
            if COUNTER >= EAR_COUNT:
                print("Drowsy!")
                send_to_cloud("Drowsy!")
                sound_warning()
                

        #If it is higher than threshold send awake and reset the counter
        else:
            print("Awake")
            send_to_cloud("Awake")
            COUNTER = 0

    
cv2.destroyAllWindows()
vs.stop()