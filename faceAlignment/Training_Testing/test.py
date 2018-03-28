import os,sys,logging
import glob
import dlib
from skimage import io

#Setting up debug states
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

# make sure there is a path to img folder in the argument
if len(sys.argv) != 3:
    logging.error( "********** Please give a path to the face images direcotry then the landmark_predictor dat for the argument! **********")
    exit()

# load the img foldfer to a variable
img_folder = sys.argv[1]

predictorPath = sys.argv[2]

testing_xml_path = os.path.join(img_folder, "testing.xml")

print("\n Testing Accuracy: {}".format( dlib.test_shape_predictor(testing_xml_path, predictorPath)))

dlib.hit_enter_to_continue()