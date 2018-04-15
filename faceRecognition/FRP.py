from PIL import Image, ImageDraw
import dlib
import numpy
import os
import os.path
from sklearn import neighbors
import pickle 

# train_dir = raw_input("----------- training folder path: ")
# model_save_path = raw_input("----------- path to save model: ")
# n_neighbors = raw_input("----------- number of neighbors to weight classification: ")

face_detector = dlib.get_frontal_face_detector()

68_landmark_predictor = 'raw_input("----------- 68 Point Predictor model path (including .dat): ")'
landmark_predictor = dlib.shape_predictor(68_landmark_predictor)

face_rec = 'raw_input("----------- face Recognition model path (including .dat): ")'
face_encoder = dlib.face_recognition_model)v1(face_recognition_model)



def images_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]

def grab_image(image_file, mode='RGB'):
    im = Image.open(image_file)
    if mode:
        im = im.convert(mode)
    return numpy.array(im)

def trim_to_bounds(css, image_shape):
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], iamge_shape[0]), max(css[3], 0)

def css_to_rect(css):
    return dlib.rectangle(css[3], css[0], css[1], css[2])

def rect_to_css(rect):
    return rect.top(), rect.right(), rect.bottom(), rect.left()

def raw_face_locations(img, num_unsample=1):
    return face_detector(img, num_unsample)

def hog(in_img, unsample=1):
    return [ trim_to_bounds(rect_to_css(face), in_img.shape) for face in raw_face_locations(img, unsample)]

def raw_face_ladmarks(face_image, face_locations=None):
    if face_locations is None:
        face_locations = raw_face_locations(face_image)
    else:
        face_locations = [css]

def face_encodings(face_img, known_faces_locations=None, jitters=1):
    raw_landmarks= raw_face_landmarks(face_image, known_faces_locations)
    return [numpy.array(face_encoder.compute_face_descriptor(face_image, raw_face_landmark_set, jitters)) for landmark_set in raw_face_ladmarks]

def train(train_dr, model_save_path=None, num_neighbors=None, knn_alg='ball_tree', verbose=True):

    X = []
    Y = []

    for class_dir in os.listdir(train_dir):
        for image_path in image_files_in_folder(os.path.join(train_dr, class_dr)):
            image = grab_image(image_path)
            face_bounding_boxes = hog(image)

            if len(face_bounding_boxes) != 1:
                if verbose:
                    print("Image {} not sutable for training: {}".format(image_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face" ))
            else:
                X.append(face_encodings(image, known_faces_locations=face_bounding_boxes)[0])
                Y.append(class_dir)
            
    if num_neighbors is None:
        num_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose numberd of neighbors myself: ", num_neighbors)

    knn_classifier = neighbors.KNeighborsClassifier(num_neighbors=num_neighbors, algorithm=knn_alg, weights='distance')
    knn_classifier.fit(X,Y)

    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_classifier, f)
    
    return knn_classifier

def recog(X_Ipath, knn_classifier=None, model_path=None, distance=0.6):
    if not os.path.isfile(X_Ipath) or os.path.spliter(X_Ipath)[1][1] not in ALLOWED_EXTENTIONS:
        raise Exception("Not a valid image path: {}".format(X_Ipath))

    if knn_classifier is None and model_path is None:
        raise Exception("You need a knn classifier for either knn_classifier or model_path")
    
    if knn_classifier is None:
        with open(model_path, 'rb') as classifier:
            knn_classifier = pickle.load(classifier)

    X_image = grab_image(X_Ipath) 
    X_face_loc = hog(X_image)

    if len(X_face_loc) == 0:
        return []

    face_encod = face_encoder(X_image, known_faces_locations=X_face_loc)

    closest = knn_classifier.kneighbors(face_encod, n_neighbors=1)
    matches = [closest[0][i][0] <= distance for i in range(len(X_face_loc))]

    return [(pre, loc) if rec else ("unknown", loc) for pre, loc, rec in zip(knn_classifier.predict(face_encod), X_face_loc, matches)]

