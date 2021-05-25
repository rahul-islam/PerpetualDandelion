import numpy as np
import cv2
import dlib
import argparse
from imutils import face_utils
from scipy.spatial import distance as dist
import tensorflow as tf
import pandas as pd

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
num_landmarks = np.array(range(68))
findCombinations(num_landmarks, len(num_landmarks), 3)
lcombs = np.asarray(lcombs, dtype=int)

cv2.namedWindow("FacePsy-Kit")

files = ['./data/param.json', './data/vaibhav.json', './data/rahul.json']

gt_df = pd.DataFrame()
for file in files:
    t_df = pd.read_json(file)
    gt_df = gt_df.append(t_df)


farr = []
larr = []
for i, gt in gt_df.iterrows():
    print(gt['fname'])
    frame = cv2.imread(gt['fname'])

#     if frame is not None:   
#         cv2.imshow("FacePsy-Kit", frame)
    h, w, c = frame.shape

    o_frame = frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape_cp = shape
        shape = face_utils.shape_to_np(shape)

        mouth = shape#[mouthStart:mouthEnd]
        print(mouth.shape)

        # face detection
        x1 = rect.left()
        y1 = rect.top()
        x2 = rect.right()
        y2 = rect.bottom()

        face = gray[y1:y2,x1:x2]
        face_color = frame[y1:y2,x1:x2]

        face = cv2.resize(face, (48, 48))
        face_color = cv2.resize(face_color, (224, 224))
        
        face = face[np.newaxis, :, :, np.newaxis]

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

        farr.append(feat)
        larr.append(gt['blow'])

        # cv2.imwrite('./processed/gray/{}.png'.format(i), face)
        cv2.imwrite('./processed/color/{}.png'.format(i), face_color)

        break  



f = np.array(farr).astype(int)
l = np.array(larr).astype(int)
import hickle as hkl
hkl.dump(f, 'data.hkl', mode='w')
hkl.dump(l, 'label.hkl', mode='w')