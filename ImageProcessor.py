from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
from PIL import Image

[print("\n") for x in range(4)]

def reduceImage(inPath, outPath):
    img = Image.open(inPath)
    img = img.resize((300,168), Image.ANTIALIAS)
    img.save(outPath)

def loadImage( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def absoluteFilePaths(directory):
    paths = []
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
             paths.append(os.path.abspath(os.path.join(dirpath, f)))
    return paths

root = os.path.dirname(os.path.abspath(__file__))
imagesPath = '{root}/images'.format(root=root)

train_dir_raw= os.path.join(imagesPath, 'TrainRaw')
validation_dir_raw = os.path.join(imagesPath, 'TestRaw')
train_dir = os.path.join(imagesPath, 'Train')
validation_dir = os.path.join(imagesPath, 'Test')

for folder in ["/Closed", "/Open"]:
    for image in os.listdir(train_dir_raw + folder):
        reduceImage(train_dir_raw + folder + "/" + image, train_dir + folder + "/" + image)
    for image in os.listdir((validation_dir_raw + folder)):
        reduceImage(validation_dir_raw + folder + "/" + image, validation_dir + folder + "/" + image)

train_closed_dir = absoluteFilePaths(os.path.join(train_dir, 'Closed'))
train_open_dir = absoluteFilePaths(os.path.join(train_dir, 'Open'))
validation_closed_dir = absoluteFilePaths(os.path.join(validation_dir, 'Closed'))
validation_open_dir = absoluteFilePaths(os.path.join(validation_dir, 'Open'))

train_closed_labels = np.array([0.0 for x in train_closed_dir])
train_open_labels = np.array([1.0 for x in train_open_dir])
validation_closed_labels = np.array([0.0 for x in validation_closed_dir])
validation_open_labels = np.array([1.0 for x in validation_open_dir])

train_closed_nparray = np.array([loadImage(x) for x in train_closed_dir])
train_open_nparray = np.array([loadImage(x) for x in train_open_dir])
validation_closed_nparray = np.array([loadImage(x) for x in validation_closed_dir])
validation_open_nparray = np.array([loadImage(x) for x in validation_open_dir])

train_labels = np.concatenate((train_closed_labels, train_open_labels), axis=0)
validation_labels = np.concatenate((validation_closed_labels, validation_open_labels), axis=0)

train_nparray = np.concatenate((train_closed_nparray, train_open_nparray), axis=0)
validation_nparray = np.concatenate((validation_closed_nparray, validation_open_nparray), axis=0)