#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# German Traffic Sign Recognition Benchmark (GTSRB)
# http://benchmark.ini.rub.de/?section=gtsrb
# J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel
#  Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition
#  Neural Networks, http://dx.doi.org/10.1016/j.neunet.2012.02.016

# Download code (c) 2017 by Thomas Viehmann <tv.code@beamnet.de>
# licensed under the MIT License

import os
from keras.utils.data_utils import get_file
from keras.utils.generic_utils import Progbar
import numpy
import zipfile
import PIL
import pandas

PATH_TRAINING     = "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip"
PATH_TEST         = "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip"
PATH_TEST_LABELS  = "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip"
PATH_ANALYSIS_SRC = "http://benchmark.ini.rub.de/Dataset/tsr-analysis-src.zip"

categories = ["20","30","50","60","70",
              "80","Ende: 80","100","120","Überholverbot",
              "Überholverbot LKW", "Vorfahrt","Vorfahrtsstraße","Vorfahrt gewähren","Halt. Vorfahrt gewähren",
              "Verbot für Fahrzeuge aller Art","Verbot für LKW","Verbot der Einfahrt","Gefahrenstelle","Kurve (L)",
              "Kurve (R)","Doppelkurve (LR)","Unebene Fahrbahn","Schleudergefahr", "Einseitig verengte Fahrbahn (R)", 
              "Arbeitsstelle","Lichtzeichenanlage","Fußgänger","Kinder","Radverkehr",
              "Eisglätte","Wildwechsel","Ende Beschränkungen","Gebot: Rechts", "Gebot: Links", 
              "Gebot: Geradeaus","Gebot: Geradeaus oder rechts", "Gebot: Geradeaus oder links","Gebot Rechts vorbei","Gebot: Links vorbei", 
              "Gebot: Kreisverkehr", "Ende Überholverbot", "Ende Überholverbot LKW"]
          
def load_cat_info():
  pasrc = get_file(os.path.basename(PATH_ANALYSIS_SRC), origin=PATH_ANALYSIS_SRC)
  images = numpy.full((43,100,100,3),255,dtype=numpy.uint8)
  with zipfile.ZipFile(pasrc) as z:
    for i in range(43):
      im = numpy.array(PIL.Image.open(z.open('resources/signs/%d.jpg'%i)))
      offset = (100-im.shape[0])//2
      images[i,offset:offset+im.shape[0]] = im
  images = numpy.array(images)
  return images

def load_data(target_size=(224,224), min_size=(1,1), verbose=False, return_batches=False):
  """
  Loads data of
  
    German Traffic Sign Recognition Benchmark (GTSRB)

    http://benchmark.ini.rub.de/?section=gtsrb

    J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel
    Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition
    Neural Networks, http://dx.doi.org/10.1016/j.neunet.2012.02.016

  # Arguments
    target_size: Size tuple to resize images to (passed to `PIL.Image.resize`)
      If `target_size` is `None`, no resizing is done and the result will be a list of images rather than an array.
      Default: (224, 224) This is the imagenet standard size.
    min_size: Minimal size of image to be considered `PIL` size convention of (width, height).
      If you want no filtering of the size, then use `(1,1)`.
    verbose: set to `True` if you want progress output (default `False`)
    return_batches: set to `True` if you want the batch numbers for the training data as well (default: `False`)
  # Returns
    (training_images, training_labels),(test_images, test_labels) Training and test data (not one-hot encoded, return_batches=`False`)
    
    or
    
    (training_images, training_labels, batch_numbers),(test_images, test_labels) Training and test data (not one-hot encoded, return_batches=`True`) 
  """
  ptraining, ptest, ptestlabels = [get_file(os.path.basename(url), origin=url) for url in (PATH_TRAINING, PATH_TEST, PATH_TEST_LABELS)]
  images = []
  labels = []
  batches = []
  if verbose: print ("Extracting training images")
  with zipfile.ZipFile(ptraining) as z:
    names = [n for n in z.namelist() if n.endswith('.ppm')]
    num = len(names)
    if verbose: pb = Progbar(num)
    for i,n in enumerate(names):
      if verbose and i % 100 == 0:
        pb.update(i)
      im = PIL.Image.open(z.open(n))
      if im.size[0]>=min_size[0] and im.size[1]>=min_size[1]:
        if target_size is not None:
          im = im.resize(target_size)
        images.append(numpy.array(im))
        labels.append(int(os.path.basename(os.path.dirname(n))))
        if return_batches:
          batches.append(int(os.path.basename(n).split('_')[0]))
    if verbose: pb.update(num, force=True)
  if target_size is not None: 
    images = numpy.array(images)
  labels = numpy.array(labels)
  
  if verbose: print ("\nExtracting test images")
  with zipfile.ZipFile(ptestlabels) as z:
    test_metadata = pandas.read_csv(z.open('GT-final_test.csv'), sep=';')
  test_metadata = test_metadata[(test_metadata["Width"]>=min_size[0])&(test_metadata["Height"]>=min_size[1])]

  num = len(test_metadata)  
  test_images = []
  test_labels = []
  if verbose: pb = Progbar(num)
  with zipfile.ZipFile(ptest) as z:
    for i,(n,l) in enumerate(zip(test_metadata["Filename"],test_metadata["ClassId"])):
      if verbose and i % 100 == 0:
        pb.update(i)
      im = PIL.Image.open(z.open('GTSRB/Final_Test/Images/%s'%n))
      if target_size is not None:
        im = im.resize(target_size)
      test_images.append(numpy.array(im))
      test_labels.append(l)
  if verbose: pb.update(num, force=True)
  if target_size is not None: 
    test_images = numpy.array(test_images)
  test_labels = numpy.array(test_labels)
  if verbose: print ("\n")
  if return_batches:
    batches = numpy.array(batches)
    return (images, labels, batches), (test_images, test_labels)
  return (images, labels), (test_images, test_labels)
  
