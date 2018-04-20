# FaceRegCapstone
Face Recognition Capstone:
-This is the Capstone program of Ian Clark and Zachary Resler

Requirements:
-preferably on linux or ios becuase some of the python pagages aren't supported on windows yet, but can easily run on windows, just make sure you have compatible versions of libraries

-16GB of RAM as some of the trianing, depending on the size of the dataset and training variables you're using, can be pretty taxing

-Ideally the faster your processor and how many cores it has will also help speed things along

-Visual C++ interpreter for python, some of the packages we are using are convertions of c++ for python

python 2.7 64bit version
    -packages:
        - dlib
        - boost
        - scikit-image
        - scikit-learn
        - numpy
        - scipy
        - cmake
        - opencv-python
        - openface 
        - pillow

-Text editor ( i.e Visual Studio, pycharm )

-CC and CXX compiler, We used MinGW64 as ours

-iBug datasets used for training and testing the landmark predictor 
(http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz)

-Pre-Trained 68 point landmark predictor used to create the landmark values for the xml generator (https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2)

-Pretrained 128 point predictor model used for getting face measurements
(https://github.com/davisking/dlib-models/raw/master/dlib_face_recognition_resnet_model_v1.dat.bz2)

Created out own training img set for the face recognition part, structure should look like this:
    -training_images
    |__ Ian Clark
    |  |__ian_clark_1.jpeg
    |  |__ian_clark_2.jpeg
    |
    |__ Gary_Busey
    |  |__gary_busey_1.jpeg
    |  |__gary_busey_2.jpeg

and so on

Should have around 20 images in each subfolder with only the face you want to be known visable.
Images too large or "awkward format will probably get discarded

