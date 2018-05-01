#################################################################################
# 
#   - This program actually takes an image folder and runs the recognition
#       classifier on them then displays the image with the names overlayed
#
#   - You'll be prompted to provide paths to the images folder and classifier
#
#   - Image folder doesn't require any stucture to be passed in
#
#################################################################################



import facePackages.FRP as FRP
import os, os.path, pickle, math
from sklearn import neighbors
from PIL import Image, ImageDraw
import dlib

print("----------- Welcome to the face recognition program!")

# Grab the file with the images to be recognised from the user
image_path = raw_input("----------- Please enter the path to the image you want to use: ")

# Get the classifier from the user
classifier = raw_input("----------- Please enter the path to the classifier you wish to use: ")

# for each image in the file recognise the faces in the image
for img in os.listdir(image_path):
    file_path = os.path.join(image_path, img)

    print("----------- Look'n for a face in: {}".format(img))

    # pass the image to be recognized
    predict = FRP.recog(file_path, model_path=classifier)

    # for every name predicted output it to the console
    for name, (top, right, bottom, left) in predict:
        print("----------- Found {} at ({}, {})".format(name, left, top))

    # Show all the face with the names attached
    FRP.show_known_face_name(file_path, predict)
    
    dlib.hit_enter_to_continue

print("----------- End!")