################################################################################
# - Used to take in a pretrained model and the images to be used to help train
#   the new model.
#
# - Creates an xml file that can be used to train or test a landmark detection 
#   model
#
# - TODO: prettify the xml so it's easier for humans to read
# - TODO: Look at multithreading so this thing doesn't run at a snails pace 
# - TODO: Fix the file naming so all unacceptable characters are caught
#
################################################################################

import sys, os, dlib, glob
import logging
from skimage import io
from xml.etree.ElementTree import Element, SubElement, Comment, tostring


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

# if the argument 
if len(sys.argv) != 1:
    print("\n arg -> python [0] ")
    exit()

# grab the landmark model and the image folder path from the user
landmark_path = raw_input("----------- Please enter the path to the landmark model: ")
images_path = raw_input("----------- Please enter the path to the images to use: ")

#initialise the face dtector and the landmark predictor 
detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(landmark_path)

# initialize the display window
win = dlib.image_window()

# for every image in the file predict the landmarks and overlay them on the image
for f in glob.glob(os.path.join(images_path, "*.jpg")):

    # read the image in
    img = io.imread(f)
  
    # set up the window with the image
    win.clear_overlay()
    win.set_image(img)

    # send the image through the detector
    det = detector(img, 1)

    # for every face in the detected faces get the landmarks
    for k, d in enumerate(det):
        
        # pass the image into the landmark model
        shape = landmark_predictor(img, d)

        # overlay the landmarks onto the image in the window
        win.add_overlay(shape)

    # overlay the bounding box    
    win.add_overlay(det)
dlib.hit_enter_to_continue()