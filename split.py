# 1. VideoCapture: Our framework uniformly extracts 120 frames from 30-th to 270-th frames of every video.

# 2. FaceDetector: Face detection is performed via CenterNet with batchsize of 60 frames. Our framework selects at most 2 faces with top detection confidence if more than 2 faces are detected within a single frame. Faces with low confidence are abandoned.

# FaceTracker: Face tracking is performed via direct IOU matching between bounding boxes from current and previous frames. Tracking sequences with less than 20  faces are ignored (most likely false detection).

# FaceClassifier: The system then predicts classification scores of all selected faces with a batch size of 120. Results of three classifiers are ensembled here for each face.

# PostProcess: The video score is calculated as a weighted average of the frame scores with weights being the detection confidences. If there are multiple tracking sequences in a video, the maximum score is used as the final prediction of the entire video.

import cv2
import numpy as np 
import os

# Capture video with OpenCV
cap = cv2.VideoCapture('teste.mp4')

try:
    if not os.path.exists('data'):
        os.makedirs('data')
except OSError:
    print("Error: Can't create data directory")

# Set currentFrame to ZERO
currentFrame = 0

while(True):

    # Capture image Frame by Frame
    ret, Frame = cap.read()

    # Save frame into folder
    name = './data/frame' + str(currentFrame) + '.jpg'
    print ('Creating...' + name)
    cv2.imwrite(name, Frame)
    
    # Next frame
    currentFrame += 1

# When everything 
cap.release()
cv2.destroyAllWindows()