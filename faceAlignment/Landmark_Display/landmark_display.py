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
#win = dlib.image_window()

for f in glob.glob(os.path.join(images_path, "*.jpg")):
    print("Processing file: {}".format(f))
    img = io.imread(f)

    #win.clear_overlay()
    #win.set_image(img)

    det = detector(img, 1)
    print("Number of faces detected: {}".format(len(det)))
    for k, d in enumerate(det):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
        shape = landmark_predictor(img, d)
        print ("Parrt 0: {}, Part 1: {} ...".format(shape.part(0),shape.part(1)))
       # win.add_overlay(shape)

    #win.add_overlay(det)
    #dlib.hit_enter_to_continue()