# TODO: Under development
# will be looking for examples 

import facePackages.FRP as FRP
import cv2

video = raw_input("----------- Enter the path to the video(including the .mp4): ")
convert_video = cv2.VideoCapture(video)
length = int(convert_video.get(cv2.CAP_PROP_FRAME_COUNT))

# create resulting video and set the frame rate and resolution of the video you're passing in
cc = cv2.VideoWriter(*'XVID')
resulting_video = cv2.VideoWriter('face_recognition_video.avi', cc, 29.90, (1920, 1080))

# load images to learn 