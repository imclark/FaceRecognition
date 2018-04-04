################################################################################
# - Used to take in a pretrained model and the images to be used to help train
#   the new model.
#
# - iBug dataset used for training and testing 
#        http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz
#
# - Pretrained shape predictor used to create the 68 point values
#       https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2
#
# - Creates an xml file that can be used to train or test a landmark detection 
#   model
#
# - TODO: prettify the xml so it's easier for humans to read
# - TODO: Look at multithreading so this thing doesn't run at a snails pace 
# - TODO: Fix the file naming so all unacceptable characters are caught
#
################################################################################

import sys, os, dlib, glob, gc
import logging
from skimage import io
from xml.etree.ElementTree import Element, SubElement, Comment, tostring

gc.enable()

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
        
        
        part = SubElement(box, 'part', name= '00', x= str(shape.part(0).x), y= str(shape.part(0).y))
        part = SubElement(box, 'part', name= '01', x= str(shape.part(1).x), y= str(shape.part(1).y))
        part = SubElement(box, 'part', name= '02', x= str(shape.part(2).x), y= str(shape.part(2).y))
        part = SubElement(box, 'part', name= '03', x= str(shape.part(3).x), y= str(shape.part(3).y))
        part = SubElement(box, 'part', name= '04', x= str(shape.part(4).x), y= str(shape.part(4).y))
        part = SubElement(box, 'part', name= '05', x= str(shape.part(5).x), y= str(shape.part(5).y))
        part = SubElement(box, 'part', name= '06', x= str(shape.part(6).x), y= str(shape.part(6).y))
        part = SubElement(box, 'part', name= '07', x= str(shape.part(7).x), y= str(shape.part(7).y))
        part = SubElement(box, 'part', name= '08', x= str(shape.part(8).x), y= str(shape.part(8).y))
        part = SubElement(box, 'part', name= '09', x= str(shape.part(9).x), y= str(shape.part(9).y))
        part = SubElement(box, 'part', name= '10', x= str(shape.part(10).x), y= str(shape.part(10).y))
        part = SubElement(box, 'part', name= '11', x= str(shape.part(11).x), y= str(shape.part(11).y))
        part = SubElement(box, 'part', name= '12', x= str(shape.part(12).x), y= str(shape.part(12).y))
        part = SubElement(box, 'part', name= '13', x= str(shape.part(13).x), y= str(shape.part(13).y))
        part = SubElement(box, 'part', name= '14', x= str(shape.part(14).x), y= str(shape.part(14).y))
        part = SubElement(box, 'part', name= '15', x= str(shape.part(15).x), y= str(shape.part(15).y))
        part = SubElement(box, 'part', name= '16', x= str(shape.part(16).x), y= str(shape.part(16).y))
        part = SubElement(box, 'part', name= '17', x= str(shape.part(17).x), y= str(shape.part(17).y))
        part = SubElement(box, 'part', name= '18', x= str(shape.part(18).x), y= str(shape.part(18).y))
        part = SubElement(box, 'part', name= '19', x= str(shape.part(19).x), y= str(shape.part(19).y))
        part = SubElement(box, 'part', name= '20', x= str(shape.part(20).x), y= str(shape.part(20).y))
        part = SubElement(box, 'part', name= '21', x= str(shape.part(21).x), y= str(shape.part(21).y))
        part = SubElement(box, 'part', name= '22', x= str(shape.part(22).x), y= str(shape.part(22).y))
        part = SubElement(box, 'part', name= '23', x= str(shape.part(23).x), y= str(shape.part(23).y))
        part = SubElement(box, 'part', name= '24', x= str(shape.part(24).x), y= str(shape.part(24).y))
        part = SubElement(box, 'part', name= '25', x= str(shape.part(25).x), y= str(shape.part(25).y))
        part = SubElement(box, 'part', name= '26', x= str(shape.part(26).x), y= str(shape.part(26).y))
        part = SubElement(box, 'part', name= '27', x= str(shape.part(27).x), y= str(shape.part(27).y))
        part = SubElement(box, 'part', name= '28', x= str(shape.part(28).x), y= str(shape.part(28).y))
        part = SubElement(box, 'part', name= '29', x= str(shape.part(29).x), y= str(shape.part(29).y))
        part = SubElement(box, 'part', name= '30', x= str(shape.part(30).x), y= str(shape.part(30).y))
        part = SubElement(box, 'part', name= '31', x= str(shape.part(31).x), y= str(shape.part(31).y))
        part = SubElement(box, 'part', name= '32', x= str(shape.part(32).x), y= str(shape.part(32).y))
        part = SubElement(box, 'part', name= '33', x= str(shape.part(33).x), y= str(shape.part(33).y))
        part = SubElement(box, 'part', name= '34', x= str(shape.part(34).x), y= str(shape.part(34).y))
        part = SubElement(box, 'part', name= '35', x= str(shape.part(35).x), y= str(shape.part(35).y))
        part = SubElement(box, 'part', name= '36', x= str(shape.part(36).x), y= str(shape.part(36).y))
        part = SubElement(box, 'part', name= '37', x= str(shape.part(37).x), y= str(shape.part(37).y))
        part = SubElement(box, 'part', name= '38', x= str(shape.part(38).x), y= str(shape.part(38).y))
        part = SubElement(box, 'part', name= '39', x= str(shape.part(39).x), y= str(shape.part(39).y))
        part = SubElement(box, 'part', name= '40', x= str(shape.part(40).x), y= str(shape.part(40).y))
        part = SubElement(box, 'part', name= '41', x= str(shape.part(41).x), y= str(shape.part(41).y))
        part = SubElement(box, 'part', name= '42', x= str(shape.part(42).x), y= str(shape.part(42).y))
        part = SubElement(box, 'part', name= '43', x= str(shape.part(43).x), y= str(shape.part(43).y))
        part = SubElement(box, 'part', name= '44', x= str(shape.part(44).x), y= str(shape.part(44).y))
        part = SubElement(box, 'part', name= '45', x= str(shape.part(45).x), y= str(shape.part(45).y))
        part = SubElement(box, 'part', name= '46', x= str(shape.part(46).x), y= str(shape.part(46).y))
        part = SubElement(box, 'part', name= '47', x= str(shape.part(47).x), y= str(shape.part(47).y))
        part = SubElement(box, 'part', name= '48', x= str(shape.part(48).x), y= str(shape.part(48).y))
        part = SubElement(box, 'part', name= '49', x= str(shape.part(49).x), y= str(shape.part(49).y))
        part = SubElement(box, 'part', name= '50', x= str(shape.part(50).x), y= str(shape.part(50).y))
        part = SubElement(box, 'part', name= '51', x= str(shape.part(51).x), y= str(shape.part(51).y))
        part = SubElement(box, 'part', name= '52', x= str(shape.part(52).x), y= str(shape.part(52).y))
        part = SubElement(box, 'part', name= '53', x= str(shape.part(53).x), y= str(shape.part(53).y))
        part = SubElement(box, 'part', name= '54', x= str(shape.part(54).x), y= str(shape.part(54).y))
        part = SubElement(box, 'part', name= '55', x= str(shape.part(55).x), y= str(shape.part(55).y))
        part = SubElement(box, 'part', name= '56', x= str(shape.part(56).x), y= str(shape.part(56).y))
        part = SubElement(box, 'part', name= '57', x= str(shape.part(57).x), y= str(shape.part(57).y))
        part = SubElement(box, 'part', name= '58', x= str(shape.part(58).x), y= str(shape.part(58).y))
        part = SubElement(box, 'part', name= '59', x= str(shape.part(59).x), y= str(shape.part(59).y))
        part = SubElement(box, 'part', name= '60', x= str(shape.part(60).x), y= str(shape.part(60).y))
        part = SubElement(box, 'part', name= '61', x= str(shape.part(61).x), y= str(shape.part(61).y))
        part = SubElement(box, 'part', name= '62', x= str(shape.part(62).x), y= str(shape.part(62).y))
        part = SubElement(box, 'part', name= '63', x= str(shape.part(63).x), y= str(shape.part(63).y))
        part = SubElement(box, 'part', name= '64', x= str(shape.part(64).x), y= str(shape.part(64).y))
        part = SubElement(box, 'part', name= '65', x= str(shape.part(65).x), y= str(shape.part(65).y))
        part = SubElement(box, 'part', name= '66', x= str(shape.part(66).x), y= str(shape.part(66).y))
        part = SubElement(box, 'part', name= '67', x= str(shape.part(67).x), y= str(shape.part(67).y))
        
    sys.stdout.write("\rProcessing image %i" % num_img)
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