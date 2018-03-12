################################################################################
# - Example of how the face detection with dlib works and looks
# 
#
#  - TODO: change the arguments handled so a folder of images can be taken in
#
################################################################################

import logging, sys
import dlib
from skimage import io

#Setting up debug states
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

 #take in the image from the user's command
file_name = sys.argv[1]
logging.debug("-Loading in image: " + file_name)

# create an instance of the HOG face deterctor
logging.debug("-Initilizing the face detector")
face_detector = dlib.get_frontal_face_detector()

# create a GUI window to view the image
logging.debug("-Initilizing GUI image display")
win = dlib.image_window()

# format the image file into an array
logging.debug("-Loading image file into an array")
image = io.imread(file_name)

# run the face detector with the image
logging.debug("-Loading the image array into the face_detector")
detected_faces = face_detector(image, 1)

# load the image into the win and set the title
logging.debug("-Loading the image array into the GUI image display")
win.set_image(image)
win.set_title("HOG Face Detector")

# loop through the detected_faces and get an face_rect instance for each face regognized in the picture and place a border around the face
logging.debug("-Overlaying borders onto the faces in the image")
for i, face_rect in enumerate(detected_faces):
	print("- Face #{} found at Left: {} Top: {} width: {} height: {}".format(i, face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()))
	win.add_overlay(face_rect)

print("\n -Found {} face(s) in the image {} \n".format(len(detected_faces), file_name))

#function to make the gui image stay up untill you dont want it up anymore     
dlib.hit_enter_to_continue()