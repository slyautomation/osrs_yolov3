import xml.etree.ElementTree as ET
from os import getcwd
import os


dataset_train = 'OID\\Dataset\\train\\'
dataset_file = 'weighted_files/4_CLASS_test.txt'
classes_file = dataset_file[:-4]+'_classes.txt'


CLS = os.listdir(dataset_train)
classes =[dataset_train+CLASS for CLASS in CLS]
wd = getcwd()


def test(fullname):
    bb = ""
    in_file = open(fullname)
    tree=ET.parse(in_file)
    root = tree.getroot()
    for i, obj in enumerate(root.iter('object')):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in CLS or int(difficult)==1:
            continue
        cls_id = CLS.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        bb += (" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

    if bb != "":
        list_file = open(dataset_file, 'a')
        file_string = str(fullname)[:-4]+'.jpg'+bb+'\n'
        list_file.write(file_string)
        list_file.close()



for CLASS in classes:
    for filename in os.listdir(CLASS):
        if not filename.endswith('.xml'):
            continue
        fullname = os.getcwd()+'\\'+CLASS+'\\'+filename
        test(fullname)

for CLASS in CLS:
    list_file = open(classes_file, 'a')
    file_string = str(CLASS)+"\n"
    list_file.write(file_string)
    list_file.close()
