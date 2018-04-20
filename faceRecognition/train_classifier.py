import facePackages.FRP as FRP
import os, os.path, pickle, math
from sklearn import neighbors
from PIL import Image, ImageDraw
import dlib

print("----------- Welcome to the classifier trainer.")

training_images = raw_input("----------- Please enter the path to the training image directory: ")

save_name = raw_input("----------- Save the classifier with a specific name with .clf at the end: ")

print("----------- Training the classifier now!")

if save_name is None:
    classifier = FRP.train(training_images, model_save_path="trained_classifier.clf", num_neighbors=2)
    print("----------- Your model has been trained: {}".format('trained_classifier.clf'))
else:
    classifier = FRP.train(training_images, model_save_path=save_name, num_neighbors=2)
    print("----------- Your model has been trained: {}".format(save_name))

dlib.hit_enter_to_continue()
