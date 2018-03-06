################################################################################
# - Will take in a training xml and a testing xml and create a landmark 
#   prediction model that will then be used for face alignment 
#
# - We are using the HELEN dataset for training and testing
#
################################################################################

import os,sys,logging
import glob
import dlib
from skimage import io

#Setting up debug states
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

# make sure there is a path to img folder in the argument
if len(sys.argv) != 2:
    logging.error( "********** Please give a path to the face images direcotry for the argument! **********")
    exit()

# load the img foldfer to a variable
img_folder = sys.argv[1]
logging.debug("-Loaded in images")

# get the taining options from the dlib folder
training_options = dlib.shape_predictor_training_options()
logging.debug("-Loaded in NN training options")

# here we set the traing oversampling size to 300 since we ahve a small sample size
# this may help!? (may also try 100 to see the diffrence in training results)

training_options.oversampling_amount = 300
logging.debug("-Set oversampling to 300")

# setting the regulization perameter to a small number since the sample size is small
# hopefully this doesn't result in underfitting (will chanfe the value to see output)
training_options.nu = 0.05
logging.debug("-Set nu to 0.05")

#set the tree depth to 2
training_options.tree_depth = 2
logging.debug("-Set tree depth to 2")

#setting the be_verbose setting to true so the training data will be printed out
training_options.be_verbose = True
logging.debug("-Such Verbose, Much Information, WOW!")

# the training will take in an xml file with the imgs used for the training dataset
# so we need to create that xml with the imgs in the given folder
training_xml_path = os.path.join(img_folder, "face_landmarks_training.xml")
logging.debug("-Created the img xml")

# now we create the predictor with the settings and imgs
dlib.train_shape_predictor(training_xml_path, "trained_landmark_predictor.dat", training_options)
logging.debug("-Trained the new predictor")

print("\n Training Accuracy: {}".format(dlib.test_shape_predictor(training_xml_path, "trained_landmark_predictor.dat")))


#now we wanna test the trained model with faces diffrent than the ones we used in training
testing_xml_path = os.path.join(img_folder, "face_landmark_testing.xml")
logging.debug("-Tested the new predictor")

print("\n Testing Accuracy: {}".format( dlib.test_shape_predictor(testing_xml_path, "trained_landmark_predictor.dat")))

dlib.hit_enter_to_continue()