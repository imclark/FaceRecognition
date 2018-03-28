import sys
from skimage import io
import os
import glob
import dlib
import cv2
import openface

predictor = "landmark_predictor.dat"

filePath = sys.argv[1]

faceDetector = dlib.get_frontal_face_detector()
facePose = dlib.shape_predictor(predictor)
faceAlign = openface.AlignDlib(predictor)

win = dlib.image_window()

for f in glob.glob(os.path.join(filePath, "*.jpg")):
    img = io.imread(f)

    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    detectedFaces = faceDetector(image, 1)

    print ("- Number of Faces: {}".format(len(detectedFaces), filePath))

    for i, faceRect in enumerate(detectedFaces):
        print("- Face #{} found at Top: {} Right: {} Bottom: {} Left: {} ".format(i, faceRect.top(), faceRect.right(), faceRect.bottom(), faceRect.left()))

        poseLandmarks = facePose(image, faceRect)

        alignedFace = faceAlign.align(534, image, faceRect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

        alignedFace = cv2.cvtColor(alignedFace, cv2.COLOR_RGB2BGR)
        
        # better to allow the images to display when there is only a few 
        # images to display, ideally only one image
        win.set_image(alignedFace)

        # this code would save the newly aligned image
        #cv2.imwrite("align_face_{}.jpg".format(i), alignedFace)

dlib.hit_enter_to_continue()