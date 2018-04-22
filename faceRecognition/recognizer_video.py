# TODO: Under development
# will be looking for examples 

import facePackages.FRP as FRP
import cv2
import dlib
import sys, io, pickle

video = raw_input("----------- Enter the path to the video(including the .mp4): ")
convert_video = cv2.VideoCapture(video)
length = int(convert_video.get(cv2.CAP_PROP_FRAME_COUNT))

# create resulting video and set the frame rate and resolution of the video you're passing in
cc = cv2.VideoWriter(*'XVID')
resulting_video = cv2.VideoWriter('face_recognition_video.avi', cc, 29.90, (1920, 1080))

# load images to learn 
ian_image = FRP.grab_image("")
ian_face_encodings = FRP.face_encodings(ian_image)[0]

# log the known faces
known_faces = [ ian_face_encodings ]

# initialize locations, encodings, names, and frame number to keep track of
face_locations = []
face_encodings = []
names = []
frame_num = 0

# loop through the video until there are no more frames
while True:

    # loop through the video and incrament the frame number
    ret, fram = convert_video.read()
    frame_num += 1

    # if out of frames then break the while loop
    if not ret:
        break
    
    # since the cv2 converts the img to BGR we need to convert it back to RGB
    converted_rgb_frame = frame[:, :, ::-1]

    # get the face locations and encodings for the img
    face_locations = FRP.hog(converted_rgb_frame)
    face_encodings = FRP.face_encodings(converted_rgb_frame, face_locations)

    # log names in img
    face_names = []

    # for each face encoding found in the image
    for face_enco in face_encodings:
        
        # log all the face encoding matches
        matches = FRP.comparison(known_faces, face_encodings, threshold=0.5)

        # initialize the name 
        name = None 

        # check matches and set the name (need to fix for multiple people)
        if match[0]:
            name = "Ian Clark"
        else:
            name = "Unkown"
        
        # add the name to the known face log
        face_names.append(name)

    # for each name get the location values
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        
        # if no name skip
        if not name:
            continue

        # overlay the name and bounding box to the frame
        # set bounding box and color
        cv2.rectangle(frame, (left, bottom -25), (right, bottom), (255, 0, 255), cv2.FILLED)
        # set the font type
        font = cv2.FONT_HERSHEY_COMPLEX
        # add the name to the frame
        cv2.putText(frame, name, (left + 6, bottom -6), font, 0.5, (255, 255, 255), 1)
    
    # print what we're doing to the console
    print("Writting to frame {} / {}".format(frame_num, length))
    
    # write to the frame
    resulting_video.write(frame)

# cleanup
convert_video.release()
cv2.destroyAllWindows()
