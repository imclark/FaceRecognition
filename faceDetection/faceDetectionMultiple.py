################################################################################
# - Example of how the face detection with dlib works and looks
# 
#
#  - TODO: change the arguments handled so a folder of images can be taken in
#
################################################################################

import logging, sys
import dlib
import glob
import os
import cv2
from skimage import io
from skimage.draw import polygon_perimeter

#Setting up debug states
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

 #take in the image from the user's command
directory_name = sys.argv[1]
#logging.debug("-Loading in image: " + file_name)

#grabs directory name
path = os.getcwd() + '/' + directory_name
logging.debug("-Initilizing the face detector")
face_detector = dlib.get_frontal_face_detector()
logging.debug("number of files to process: {} ".format(len([name for name in glob.glob(os.path.join(path, '*'))])))

#loops through each image
for filename in glob.glob(os.path.join(path, '*')):
	logging.debug("Processing file: {}".format(filename))
	image = io.imread(filename)
	#checks for color images, if color needs to shift color from BGR, TO RGB
	if(len(image.shape)==3):
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	#detect faces
	detected_faces = face_detector(image, 1)
	#building a path location of images
	base_dir = os.path.join(os.getcwd(), directory_name , directory_name + "processed")
	#creates a new directory for processed images
	if not os.path.exists(base_dir):
		os.makedirs(base_dir)
	#cycles through the found faces, and puts a box around each one
	for k, d in enumerate(detected_faces):
		logging.debug("Processing face {} ".format(k) + "for file {}".format(filename))
		logging.debug("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
		#drawing the rectangle
		cv2.rectangle(image, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 2)

	#building path to save file
	new_image_loc = base_dir + "{}".format(filename)["{}".format(filename).rfind("/"):"{}".format(filename).rfind(".")] + "faceboxed" + "{}".format(filename)["{}".format(filename).rfind("."):]
	logging.debug(base_dir)
	#saves new images
	#NOTE: DOES NOT OVERWRITE
	cv2.imwrite(new_image_loc , image)

		

		

# create an instance of the HOG face deterctor


# create a GUI window to view the image
#logging.debug("-Initilizing GUI image display")
#win = dlib.image_window()

# format the image file into an array
#logging.debug("-Loading image file into an array")


# run the face detector with the image
#logging.debug("-Loading the image array into the face_detector")
#detected_faces = face_detector(image, 1)

# load the image into the win and set the title
#logging.debug("-Loading the image array into the GUI image display")
#win.set_image(image)
#win.set_title("HOG Face Detector")

# loop through the detected_faces and get an face_rect instance for each face regognized in the picture and place a border around the face
#logging.debug("-Overlaying borders onto the faces in the image")
#for i, face_rect in enumerate(detected_faces):
#	win.add_overlay(face_rect)


#function to make the gui image stay up untill you dont want it up anymore     
dlib.hit_enter_to_continue()



