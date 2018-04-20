################################################################################
# - Takes in a the image directory in the command prompt. Then asked to provide
#   predictor model. It'll create the face detector, land mark predictor
#   and face aligner. Then uses the face detector, landmark predictor and 
#   the face aligner to generate the representations to be passed into the
#   face recognition part of the project
#
#   
#
# 
# 
# 
#
################################################################################


import sys
from skimage import io
import os
import glob
import dlib
import cv2
import openface

if len(sys.argv) != 1:
    print("\n arg -> python [0] ")
    exit()

# get the predictor model from the user
predictor = raw_input("----------- Enter the predictor model name (including .dat): ")

# get the training folder from the user
filePath = raw_input("----------- Enter the file directory path: ")

# initialize the face dtector, landmark predictor, and face aligner
faceDetector = dlib.get_frontal_face_detector()
facePose = dlib.shape_predictor(predictor)
faceAlign = openface.AlignDlib(predictor)

# initialize the display window
#win = dlib.image_window()

# for each image in the folder
for f in glob.glob(os.path.join(filePath, "*.jpg")):
    
    # convert the image into a usable array for the predictors
    img = io.imread(f)
    
    # convert the img array to a usable color image for openface
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # detect the face
    detectedFaces = faceDetector(image, 1)

    # print out the number of faces
    print ("- Number of Faces: {}".format(len(detectedFaces), filePath))

    # for each face in the image
    for i, faceRect in enumerate(detectedFaces):
    
        # print out the the face loaction
        print("- Face #{} found at Top: {} Right: {} Bottom: {} Left: {} ".format(i, faceRect.top(), faceRect.right(), faceRect.bottom(), faceRect.left()))

        # predict the landmarks of the image
        poseLandmarks = facePose(image, faceRect)

        # then using the landmarks align the face
        alignedFace = faceAlign.align(534, image, faceRect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

        # conver the image back to original color so it can be displayed in the win
        #alignedFace = cv2.cvtColor(alignedFace, cv2.COLOR_RGB2BGR)
        
        # better to allow the images to display when there is only a few 
        # images to display, ideally only one image
        #win.set_image(alignedFace)

        # this code would save the newly aligned image
        cv2.imwrite("align_face_{}.jpg".format(i), alignedFace)

dlib.hit_enter_to_continue()