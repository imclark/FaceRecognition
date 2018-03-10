################################################################################
# - Example of how the face detection with dlib works and looks
# usage python faceDetectionMultiThread.py directory_name num_of_cores
#
#  - TODO: change the arguments handled so a folder of images can be taken in
#
################################################################################

import logging, sys
import dlib
import glob
import os
import cv2
import gc
from skimage import io
from skimage.draw import polygon_perimeter
import multiprocessing
from multiprocessing import Pool





#Setting up debug states
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

 #take in the image from the user's command
directory_name = sys.argv[1]
num_procs = int(sys.argv[2])
#logging.debug("-Loading in image: " + file_name)

#grabs directory name
path = os.getcwd() + '/' + directory_name
logging.debug("-Initilizing the face detector")
face_detector = dlib.get_frontal_face_detector()
logging.debug("number of files to process: {} ".format(len([name for name in glob.glob(os.path.join(path, '*'))])))





files = glob.glob(os.path.join(path, '*'))
#loops through each image


def detectFaces(x):
	logging.debug("Processing file: {}".format(x))
	image = io.imread(x)
	if(len(image.shape)==3):
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	detected_faces = face_detector(image, 1)
	for k,d in enumerate(detected_faces):
		logging.debug("Processing face {} ".format(k) + "for file {}".format(x))
		logging.debug("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
		#drawing the rectangle
		cv2.rectangle(image, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 2)	
	new_image_loc = base_dir + "{}".format(x)["{}".format(x).rfind("/"):"{}".format(x).rfind(".")] + "faceboxed" + "{}".format(x)["{}".format(x).rfind("."):]
	logging.debug(base_dir)
	#saves new images
	#NOTE: DOES NOT OVERWRITE
	cv2.imwrite(new_image_loc , image)
	del image
	del detected_faces
	del new_image_loc
	gc.collect()
	return
	
def mp_worker(files):
	if(multiprocessing.cpu_count() > num_procs):
		p = multiprocessing.Pool(num_procs)
		p.map(detectFaces, files)
	else:
		p = multiprocessing.Pool(1)
		p.map(detectFaces, files)
		
	p.close()
	p.join()
	return
if __name__ == '__main__':

			
	base_dir = os.path.join(os.getcwd(), directory_name , directory_name + "processed")
		
	if not os.path.exists(base_dir):
		os.makedirs(base_dir)
	
	mp_worker(files)
		
	dlib.hit_enter_to_continue()



