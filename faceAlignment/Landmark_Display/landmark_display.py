################################################################################
# - Used to take in a pretrained model and the images to be used to help train
#   the new model.
#
# - Spits out all the info one would need to enter into the training and testing 
#   xml file
#
################################################################################

import sys, os, dlib, glob
import logging
from skimage import io

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

if len(sys.argv) != 3:
    print("\n arg -> python [0] program file name [1] landmark predictor [2] image directory")
    exit()

landmark_path = sys.argv[1]
images_path = sys.argv[2]

detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(landmark_path)

for f in glob.glob(os.path.join(images_path, "*.jpg")):
    print("Processing file: {}".format(f))
    img = io.imread(f)

    det = detector(img, 1)
    print("Number of faces detected: {}".format(len(det)))
    for k, d in enumerate(det):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
        shape = landmark_predictor(img, d)
        print ("Parrt 0: {}, Part 1: {} ...".format(shape.part(0),shape.part(1)))

    dlib.hit_enter_to_continue()