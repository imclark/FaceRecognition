############################################################################
# 
#   -This is a file to call to create and train a classifier
#
#   -Make sure you have the correct training image set-up
#   ex:
#       -training-images
#       |__person_1
#       |  |__person_1_image_1.jpg
#       |  |__person_1_image_2.jpg
#       |  .
#       |__person_2
#       |  |__person_2_image_1.jpg
#       |  |__person_2_image_2.jpg
#       |  .
#
#   -You'll be prompted to give paths to the specified files
#
#   -For naming of created classifier, if no name is given then a default
#       name will be provided: 'trained_classifier.clf'
#
############################################################################

import facePackages.FRP as FRP
import os, os.path, pickle, math
from sklearn import neighbors
from PIL import Image, ImageDraw
import dlib

print("----------- Welcome to the classifier trainer.")

# get training image path from the user
training_images = raw_input("----------- Please enter the path to the training image directory: ")

# get the name to save the file to
save_name = raw_input("----------- Save the classifier with a specific name (include .clf at the end): ")

print("----------- Training the classifier now!")

# if there is no save name then provide one
if save_name is None:
    classifier = FRP.train(training_images, model_save_path="trained_classifier.clf", num_neighbors=2)
    print("----------- Your model has been trained: {}".format('trained_classifier.clf'))
# else use the given name
else:
    classifier = FRP.train(training_images, model_save_path=save_name, num_neighbors=2)
    print("----------- Your model has been trained: {}".format(save_name))

dlib.hit_enter_to_continue()
