import sys, os, dlib, glob
from skimage import io 

if len(sys.argv) != 1:
    print(
        "Call the program like this: \n"
        " python [0]faceReg.py [1]lanmark_predictor.dat\n"
    )
    exit()

# input the paths to the models and img folders
landmark_pridictor = raw_input("----------- Predictor model name (including .dat): ")
face_recognition = raw_input("----------- Recognition model name (including .dat): ")
face_folder = raw_input("----------- Face folder name: ")

# initilize the predictors
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(landmark_pridictor)
faceRec = dlib.compute_face_descriptor(face_recognition)

# intitilize the gui window
win = dlib.image_window()

# grab the images from the img folder and pass each img through our system
for f in glob.glob(os.path.join(face_folder, "*.jpeg")):
    
    # print out the image being processed
    print("-----------Going through file number: {}".format(f))

    # convert the img into a useable form
    img = io.imread(f)

    # claer the image and set the image into the window
    win.clear_overlay()
    win.set_image(img)

    # pass the img through the face detector
    dets = detector(img, 1)

    # print out the number of faces found
    print("-----------Number of faces found: {}".format(len(dets)))

    # if there are no faces in the image then skip to the next image
    if len(dets) == 0:
        print("There were no faces in that image")
    # if there is at least 1 face, then proceed 
    else:
        for k, d in enumerate(dets):
            
            # Prints out the box imfo
            #print("----------- Found {}: Top: {} Right: {} Bottom: {} Left: {}".format(k, d.top(), d.right(), d.bottom(), d.left()))

            # pass the face into the landmark predictor
            shape = sp(img, d)

            # clear the overlay then add the bounded box and landmark to the image
            win.clear_overlay()
            win.add_overlay(d)
            win.add_overlay(shape)

            # start the recogniton part and print the vector result
            # you could also run without the 10 at the end to produce a faster process
            # with the 10 the process runs x10 times slower though it raises the accuracy a bit
            face_descripter = faceRec.compute_face_descriptor(img, shape, 10)
            print(face_descriptor)

            dlib.hit_enter_to_continue()


