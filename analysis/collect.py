# import numpy as np
# import cv2 as cv
# cap = cv.VideoCapture(0)
# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()
# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     # if frame is read correctly ret is True
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#     # Our operations on the frame come here
#     # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     # Display the resulting frame
#     cv.imshow('frame', frame)
#     if cv.waitKey(1) == ord('q'):
#         break
# # When everything done, release the capture
# cap.release()
# cv.destroyAllWindows()

#draw either rectangles or circles by dragging the mouse like we do in Paint application

import cv2
import numpy as np
import time
import random
import os
# import the necessary packages
import argparse
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--pname", required=True,
	help="name of the user")
args = vars(ap.parse_args())

pname = args['pname']

if not os.path.exists('./data/{}'.format(pname)):
    os.makedirs('./data/{}'.format(pname))

# print(img.shape)
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
# cv2.setWindowProperty('frame',  cv2.WINDOW_FULLSCREEN)
# cv2.setMouseCallback('image',draw_circle)s

color = (0, 255, 0)
thickness = 9
width = 3072
height = 1920



w , h = np.int(width/3), np.int(height/3)

quads = [
    [(0, 0), (w, h)],
    [(w, 0), (w*2, h)],
    [(w*2, 0), (w*3, h)],

    [(0, h), (w, h * 2)],
    [(w, h), (w*2, h * 2)],
    [(w*2, h), (w*3, h * 2)],

    [(0, h*2), (w, h * 3)],
    [(w, h*2), (w*2, h * 3)],
    [(w*2, h*2), (w*3, h * 3)],            
]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

start_time = time.time()
drawing = False

capturing = False
iter = 1
gt_json = []
while(1):
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    img = np.zeros((1920, 3072, 3), np.uint8)
    img = cv2.line(img, (0, h), (width, h), color, thickness)
    img = cv2.line(img, (0, h * 2), (width, h * 2), color, thickness)

    img = cv2.line(img, (w, 0), (w, height), color, thickness)
    img = cv2.line(img, (w * 2, 0), (w * 2, height), color, thickness)

    end_time = time.time()

    time_elapsed = (end_time - start_time)
    if time_elapsed > 1 and drawing:
        start_time = time.time()
        drawing = False
    elif time_elapsed > 5 and drawing==False:
        q = random.choice([0, 1, 2 , 3, 4, 5, 6, 7, 8])
        quad = quads[q]

        start_time = time.time()
        drawing = True
    
    if drawing:
        img = cv2.rectangle(img, quad[0], quad[1], color, -1)

    if capturing:
        img = cv2.putText(img, 'Capturing', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, color, thickness, cv2.LINE_AA)
        
        cv2.imwrite('./data/{}/{}.png'.format(pname, iter), frame)
        if drawing:
            gt_json.append({
                'fname': './data/{}/{}.png'.format(pname, iter),
                'quad': q,
                'blow': True
            })
        else:
            gt_json.append({
                'fname': './data/{}/{}.png'.format(pname, iter),
                'quad': -1,
                'blow': False
            }) 

        iter += 1



    cv2.imshow('image', img)
    if cv2.waitKey(1) == ord('c'):
        capturing = True
    if cv2.waitKey(1) == ord('q'):
        break
import json
with open('./data/{}.json'.format(pname), 'w') as outfile:
    json.dump(gt_json, outfile)

cv2.destroyAllWindows()