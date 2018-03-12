################################################################################
# - Used to take in a pretrained model and the images to be used to help train
#   the new model.
#
# - Creates an xml file that can be used to train or test a landmark detection 
#   model
#
# - TODO: prettify the xml so it's easier for humans to read
# - TODO: Look at multithreading so this thing doesn't run at a snails pace 
# - TODO: Fix the file naming so all unacceptable characters are caught
#
################################################################################

import sys, os, dlib, glob
import logging
from skimage import io
from xml.etree.ElementTree import Element, SubElement, Comment, tostring


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


# creating the dataset structure

dataset = Element('dataset')
name = SubElement(dataset, 'name')
name.text = 'training/testing data'
comment = SubElement(dataset, 'comment')
comment.test = 'Images used are from the HELEN dataset'
images = SubElement(dataset, 'images')

if len(sys.argv) != 3:
    print("\n arg -> python [0] program file name [1] landmark predictor [2] image directory")
    exit()

landmark_path = sys.argv[1]
images_path = sys.argv[2]

detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(landmark_path)

print("\nDepending on the number of images in the given folder, generating yor xml file may take a few minutes.")
print ("The program will ask you to name the xml file once it's done processing the images.\n")

num_img = 1

for f in glob.glob(os.path.join(images_path, "*.jpg")):

    img = io.imread(f)

    image = SubElement(images, 'image', file= os.path.abspath(f) )

    det = detector(img, 1)

    for k, d in enumerate(det):
        
        box = SubElement(image, 'box', top = str(d.top()), left= str(d.left()), width= str(d.width()), height= str(d.height()))
        shape = landmark_predictor(img, d)
        
        for n in range(68):
            part = SubElement(box, 'part', name= str(n), x= str(shape.part(n).x), y= str(shape.part(n).y))
        
    sys.stdout.write("\rProcessing image %i of 500 images" % num_img)
    num_img += 1


mydata = tostring(dataset)
xml_name = raw_input("xml file name (excluding .xml): ")
if len(xml_name) == "" or "!" in xml_name or "." in xml_name or "/" in xml_name:
    print("not an acceptable file name")
    exit()
xml_name += ".xml"
myfile = open(xml_name, "w")
myfile.write('<?xml version="1.0" encoding=encoding="ISO-8859-1"?>')
myfile.write('<?xml-stylesheet type="text/xsl" href="image_metadata_stylesheet.xsl"?>')
myfile.write(mydata)