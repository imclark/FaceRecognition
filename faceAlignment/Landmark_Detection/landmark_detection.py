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

if len(sys.argv) != 3:
    print("\n arg -> python [0] program file name [1] landmark predictor [2] image directory")
    exit()

landmark_path = sys.argv[1]
images_path = sys.argv[2]

detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(landmark_path)

win = dlib.image_window()

for f in glob.glob(os.path.join(images_path, "*.jpg")):

    img = io.imread(f)
  
    win.clear_overlay()
    win.set_image(img)

    det = detector(img, 1)

    for k, d in enumerate(det):
        
        shape = landmark_predictor(img, d)

        win.add_overlay(shape)
        
    win.add_overlay(det)
dlib.hit_enter_to_continue()