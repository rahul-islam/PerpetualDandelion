__author__ = "Rahul Islam"
__email__ = "rahulislam3@gmail.com"

import numpy as np
import cv2
import argparse
import os
import sys
import dlib
import argparse
from imutils import face_utils
from scipy.spatial import distance as dist
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()


import lightgbm as lgb
bst = lgb.Booster(model_file='blow_lgb_dart.txt')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./models/dlib/shape_predictor_68_face_landmarks.dat')

(mouthStart, mouthEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

def findCombinations(A, n, k, out="", i=0):
 
    # invalid input
    if k > n:
        return
 
    # base case: combination size is `k`
    if k == 0:
#         print(out)
        lcombs.append(out.strip().split(' '))
        return
 
    # start from the next index till the last index
    for j in range(i, n):
 
        # add current element `A[j]` to the solution and recur for next index
        # `j+1` with one less element `k-1`
        findCombinations(A, n, k - 1, f"{out} {A[j]}", j + 1)

def getIVA(a, b, c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle).round(2)

lcombs = []
num_landmarks = np.array(range(20))
findCombinations(num_landmarks, len(num_landmarks), 3)
lcombs = np.asarray(lcombs, dtype=int)

cv2.namedWindow("FacePsy-Kit")
cap = cv2.VideoCapture(args.cam_id) # default value
if args.file is not None:
    cap = cv2.VideoCapture(args.file)
else:
    cap = cv2.VideoCapture(args.cam_id)


### Saving Video to file
frame_width = 1280
frame_height = 720
cap.set(3, frame_width)
cap.set(4, frame_height)

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
import os

rval, frame = cap.read()

import tensorflow as tf
filename = "blow.json"

with open(filename, "r") as json_file:
    loaded_model_json = json_file.read()
    net = tf.keras.models.model_from_json(loaded_model_json)
    net.load_weights("blow.h5")

def drawExpression(frame, face):
    # face = cv2.resize(face, (224, 224))
    face = face.reshape(1,224,224,3)
    exprsn = net.predict(face)

    cv2.putText(frame, "{}   {}".format(exprsn, np.argmax(exprsn)), (x1, y1), 0, 1, (255, 255, 0), 1)


    return frame

import time
import collections
class FPS:
    def __init__(self,avarageof=50):
        self.frametimestamps = collections.deque(maxlen=avarageof)
    def __call__(self):
        self.frametimestamps.append(time.time())
        if(len(self.frametimestamps) > 1):
            return len(self.frametimestamps)/(self.frametimestamps[-1]-self.frametimestamps[0])
        else:
            return 0.0

fps = FPS()   

mov = cv2.VideoCapture('dand.mp4')
ret, dandFrame = mov.read()
blow = False

while True:

    if dandFrame is not None:   
        cv2.imshow("FacePsy-Kit", dandFrame)
        # out.write(frame)
    if blow:
        ret, dandFrame = mov.read()

    if blow == False:
        rval, frame = cap.read()

        h, w, c = frame.shape

        o_frame = frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape_cp = shape
            shape = face_utils.shape_to_np(shape)

            mouth = shape[mouthStart:mouthEnd]

            # face detection
            x1 = rect.left()
            y1 = rect.top()
            x2 = rect.right()
            y2 = rect.bottom()

            face = gray[y1:y2,x1:x2]
            face_color = frame[y1:y2,x1:x2]
            face_color = cv2.resize(face_color, (224, 224))
            
            feat = []
            for index, lcomb in enumerate(lcombs):
                a = mouth[lcomb[1]]
                b = mouth[lcomb[0]]
                c = mouth[lcomb[2]]

                t = getIVA(a, b, c)
                feat.append(t)
                
                t = getIVA(c, a, b)
                feat.append(t)

                t = getIVA(b, c, a)
                feat.append(t)

            feat = np.array([feat]).astype(int)
            pred = bst.predict(feat)
            y_pred = pred > 0.20 
            print(y_pred, pred)

            if y_pred[0]:
                blow = True
                cv2.putText(frame, "Blow", (x1, y1), 0, 1, (255, 255, 0), 1)
            else:
                cv2.putText(frame, "No Blow", (x1, y1), 0, 1, (255, 255, 0), 1)
            # frame = drawExpression(frame, face_color)
            frame = cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)               


            # estimateFaceLandmarkTFLite(frame, face_color)

        cv2.putText(frame, "FPS: {:.2f}".format(fps()), (frame_width - 180, 30), 0, 1, (255, 255, 0), 2)

    if not rval:
        raise IOError("webcam failure")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(1) & 0xFF == ord('r'):
        print('reset')
        mov = cv2.VideoCapture('dand.mp4')
        ret, dandFrame = mov.read()
        blow = False


cap.release()
# out.release()
print('Exit!')