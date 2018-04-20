import facePackages.FRP as FRP
import os, os.path, pickle, math
from sklearn import neighbors
from PIL import Image, ImageDraw
import dlib

print("----------- Welcome to the face recognition program!")

image_path = raw_input("----------- Please enter the path to the image you want to use: ")

classifier = raw_input("----------- Please enter the path to the classifier you wish to use: ")

for img in os.listdir(image_path):
    file_path = os.path.join(image_path, img)

    print("----------- Look'n for a face in: {}".format(img))

    predict = FRP.recog(file_path, model_path=classifier)

    for name, (top, right, bottom, left) in predict:
        print("----------- Found {} at ({}, {})".format(name, left, top))

    FRP.show_known_face_name(os.path.join(image_path, img), predict)
    dlib.hit_enter_to_continue

print("----------- That's all! FUUUUUUUUUUUUUUUUUUCK YEAH!!!")