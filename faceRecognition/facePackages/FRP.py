from PIL import Image, ImageDraw
import dlib
import numpy, math
import os
import os.path
from sklearn import neighbors
import pickle 
import importlib
import re
import cv2

# get the face detector
face_detector = dlib.get_frontal_face_detector()

# change import path to the path of your models
landmark_predictor = dlib.shape_predictor('landmark_model.dat')
face_encoder = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

accuracy = []


# takes in a folder path and goes through it and returns the images in it that match the allowed types of images
def images_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]

# takes in a image file and converts it into a numpy array
def grab_image(image_file, mode='RGB'):
    im = Image.open(image_file)
    if mode:
        im = im.convert(mode)
    return numpy.array(im)

# crops the image done to the detected face box
def trim_to_box(css, image_shape):
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)

# converts a tuple to a dlib rect object and returns the rect object
def css_to_rect(css):
    return dlib.rectangle(css[3], css[0], css[1], css[2])

# converts a dlib rect object to a tupple and returns the tuple
def rect_to_css(rect):
    return rect.top(), rect.right(), rect.bottom(), rect.left()

# takes a image and sets the unsample (# of times it uses it) and 
# returns the raw values straight from the dlib function
def raw_face_locations(img, num_unsample=1):
    return face_detector(img, num_unsample)

# returns a detected faces in an image
def hog(in_img, unsample=1):
    return [ trim_to_box(rect_to_css(face), in_img.shape) for face in raw_face_locations(in_img, unsample)]

# takes the image and returns the 68 face landmark values from the dlib function
def raw_face_landmarks(face_image, face_locations=None):
    if face_locations is None:
        face_locations = raw_face_locations(face_image)
    else:
        face_locations = [css_to_rect(locations) for locations in face_locations]

    predictor = landmark_predictor

    return [predictor(face_image, locations) for locations in face_locations]

# retruns the 128 face encodings for the image given
def face_encodings(face_img, known_faces_locations=None, jitters=1):
    raw_landmarks= raw_face_landmarks(face_img, known_faces_locations)
    return [numpy.array(face_encoder.compute_face_descriptor(face_img, raw_face_landmark_set, jitters)) for raw_face_landmark_set in raw_landmarks]

# returns a dictionary of face locations for every face in the images
def face_landmarks(face_image, face_locations=None):
    landmarks = raw_face_landmarks(face_image, face_locations)
    landmark_tuple = [[(p.x, p.y) for p in point.parts()] for point in landmarks]

    return [{ "chin": points[0:17], "left_eyebrow": points[17:22], "right_eyebrow": points[27:31], "nose_bridge": points[31:36], "left_eye": points[36:42], "right_eye": points[42:48], "top_lip": points[48:55] + [points[64]] + [points[63]] + [points[62]] + [points[61]] + [points[60]], "bottom_lip": points[54:60] + [points[48]] + [points[60]] + [points[67]] + [points[66]] + [points[65]] + [points[64]] } 
    for points in landmark_tuple]

# Given the face encodings it comapres them and measures the distances with the face_to_compare
# the distance tells you how well know the face is known
def distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return numpy.empty(0)

    return numpy.linalg.norm(face_encodings = face_to_compare, axis=1)

# compares known encodings to the given image and compares their measurements to see if their similar
def comparison(known_face_encodings, face_to_check, threshold=0.6):
    return list(distance(known_face_encodings, face_to_check) <= threshold)

# creates a directory for knwon faces and saves the knwon faces
def save_images(img_name, new_image):
    # create the known faces directory if not already created
    if not (os.path.exists("known_faces")):
        os.makedirs("known_faces")
    
    # saves the image with a new name in the known faces directory
    new_image.save('known_faces/Known_{}.jpg'.format(os.path.split(img_name)[1]))

# this is the training function that takes in the training images and produces an classifier that "learns" the faces given
# uses the N_Neighbor method for classification and the ball tree alg
def train(train_dr, model_save_path=None, num_neighbors=None, knn_alg='ball_tree'):

    X = []
    Y = []

    # go through each person in the folder
    for class_dir in os.listdir(train_dr):
        # go through each image in the person's folder
        for image_path in images_in_folder(os.path.join(train_dr, class_dir)):
            image = grab_image(image_path)
            face_bounding_boxes = hog(image)

            # if there is no one in the photo then skip it
            if len(face_bounding_boxes) != 1:
                print("Image {} not sutable for training: {}".format(image_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face" ))
            else:
                # else pass the encoding info
                X.append(face_encodings(image, known_faces_locations=face_bounding_boxes)[0])
                Y.append(class_dir)

    # set the neighbor amount        
    if num_neighbors is None:
        num_neighbors = int(round(math.sqrt(len(X))))
        print("Chose number of neighbors myself: ", num_neighbors)

    # make and train the new classifier
    knn_classifier = neighbors.KNeighborsClassifier(n_neighbors=num_neighbors, algorithm=knn_alg, weights='distance')
    knn_classifier.fit(X,Y)

    # now save the classifier with the specified path
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_classifier, f)
    
    # return the classifier
    return knn_classifier

# Recognises a face using a created classifier (the distance is the threshold for allowing a face to be classified)
# for an unknown face unknown will be set as the name
def recog(X_Ipath, knn_classifier=None, model_path=None, distance=0.6):
    
    if not os.path.isfile(X_Ipath) or os.path.splitext(X_Ipath)[1][1:] not in ('png', 'jpg', 'jpeg'):
        raise Exception("Not a valid image path: {}".format(X_Ipath))

    if knn_classifier is None and model_path is None:
        raise Exception("You need a knn classifier for either knn_classifier or model_path")
    
    # Load the classifier
    if knn_classifier is None:
        with open(model_path, 'rb') as classifier:
            knn_classifier = pickle.load(classifier)

    # grab the image and find the face info
    X_image = grab_image(X_Ipath) 
    X_face_loc = hog(X_image)

    if len(X_face_loc) == 0:
        return []

    # pass the face locations and the image to the recogniser
    face_encod = face_encodings(X_image, known_faces_locations=X_face_loc)

    # now use the classifier to find the clossest match and get the name
    closest = knn_classifier.kneighbors(face_encod, n_neighbors=1)
    matches = [closest[0][i][0] <= distance for i in range(len(X_face_loc))]

    # Bring in the accuracy variable
    global accuracy

    # if there is a match and it's within the threshold, add it to the list
    if matches and closest[0][0][0] <= distance:
        accuracy.append([knn_classifier.predict(face_encod), 1-closest[0][0][0]])

    return [(pre, loc) if rec else ("unknown", loc) for pre, loc, rec in zip(knn_classifier.predict(face_encod), X_face_loc, matches)]

# Displays the image with the name added to the bounding box
def show_known_face_name(image_path, predictions):
   
   # convert ot a usable image then add the image
    the_image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(the_image)

    # initialize the accuracy variable and creat increment counter
    global accuracy
    i = 0
    # for each name predicted
    for name, (top, right, bottom, left) in predictions:
        
        # draw the bounding box
        draw.rectangle(((left, top), (right, bottom)), outline=(25, 25, 122))

        # big 'ol bug so the text needs to be something specific
        name = name.encode("UTF-8")

        # now set the dimenstions for the name to be added to the bounding box
        height, width = draw.textsize(name)
        draw.rectangle(((left, bottom-height-10), (right, bottom)), fill=(25,25,122), outline=(25,25,122))

        # add the accuracy rating to the name to be displayed
        # if the list is not null
        if accuracy:
            # if the name associated with the accracy is the name being writen now
            if str(accuracy[i][0][0]) == name:
                # round the accuracy to the nearest hundrath decimal
                rounded = round(accuracy[i][1], 2)
                #then creat a string with a normalized accuracy rating
                acc = "  Accuracy: " + str(100*rounded) + " %"
                #add the accuracy string to the name
                name += acc

        draw.text((left+5, bottom-height-5), name, fill=(255,255,255, 255))

        i += 1

    # Shows the image with the new boudning box and prediction name
    the_image.show()

    save = raw_input("----------- Do you want to save this image? (y/n): ")

    # while we don't get a proper response ask them again and again
    while (save.lower() not in ('y', 'n', 'yes', 'no')):
        save = raw_input("----------- Please enter a valid respose to save this image or not? (y/n): ")
    
    # if they do want to save the image then save it to the default file
    if(save.lower() in ('y', 'yes')):
        print("----------- Saving image {}".format(os.path.split(image_path)[1]))
        save_images(image_path, the_image)
    
    # just cleans up the memory usage a bit
    del draw