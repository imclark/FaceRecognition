import logging, sys
import dlib
from skimage import io

#Setting up debug states
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

 #take in the image from the user's command
file_name = sys.argv[1]
logging.debug("Loading in image: " + file_name)

# create an instance of the HOG face deterctor
face_detector = dlib.get_frontal_face_detector()
logging.debug("initilizing the face detector")

# create a GUI window to view the image
win = dlib.image_window()

# format the image file into an array
image = io.imread(file_name)

# run the 
detected_faces = face_detector(image, 1)

print("Found {} faces in the image {}".format(len(detected_faces), file_name))


win.set_image(image)
win.set_title("HOG Face Detector")


for i, face_rect in enumerate(detected_faces):

	print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

	win.add_overlay(face_rect)
     
dlib.hit_enter_to_continue()