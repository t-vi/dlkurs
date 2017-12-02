#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# German Traffic Sign Recognition Benchmark (GTSRB)
# http://benchmark.ini.rub.de/?section=gtsrb
# J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel
#  Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition
#  Neural Networks, http://dx.doi.org/10.1016/j.neunet.2012.02.016

# Download code (c) 2017 by Thomas Viehmann <tv.code@beamnet.de>
# licensed under the MIT License

from __future__ import print_function
import torch.utils.data as data

import os
import numpy
import zipfile
import PIL.Image
import pandas

URL_TRAINING     = "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip"
URL_TEST         = "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip"
URL_TEST_LABELS  = "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip"
URL_ANALYSIS_SRC = "http://benchmark.ini.rub.de/Dataset/tsr-analysis-src.zip"

class GTSRBDataset(data.Dataset):
    def __init__(self, im, lab, transform, target_transform):
        self.im = im
        self.lab = lab
        self.transform = transform
        self.target_transform = target_transform
        super(GTSRBDataset, self).__init__()
    def __len__(self):
        return len(self.lab)
    def __getitem__(self, index):
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = PIL.Image.fromarray(self.im[index], mode='RGB')
        target = int(self.lab[index])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class GTSRBSource:
    urls = [
        URL_TRAINING,
        URL_TEST,
        URL_TEST_LABELS,
        URL_ANALYSIS_SRC,
    ]
    raw_folder = 'GTSRB_raw'

    def __init__(self, root, transform=None, target_transform=None, split_validation=0.1, download=False, target_size=(224,224), min_size=(1,1)):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.target_size = target_size
        self.min_size = min_size
        self.split_validation = split_validation

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        self.load()
        self.load_cat_info()

    def train(self):
        return GTSRBDataset(self.images, self.labels, self.transform, self.target_transform)

    def test(self):
        return GTSRBDataset(self.test_images, self.test_labels, self.transform, self.target_transform)

    def val(self):
        return GTSRBDataset(self.val_images, self.val_labels, self.transform, self.target_transform)

    def _check_exists(self):
        ptraining   = os.path.join(self.root, self.raw_folder, os.path.basename(URL_TRAINING))
        ptest       = os.path.join(self.root, self.raw_folder, os.path.basename(URL_TEST))
        ptestlabels = os.path.join(self.root, self.raw_folder, os.path.basename(URL_TEST_LABELS))
        return (os.path.exists(ptraining) and os.path.exists(ptest) and os.path.exists(ptestlabels))

    def get_cat_info(self):
        return self.cat_images

    def load_cat_info(self):
      pasrc   = os.path.join(self.root, self.raw_folder, os.path.basename(URL_ANALYSIS_SRC))
      if not os.path.exists(pasrc):
          raise RuntimeError('Dataset not found.' +
                             ' You can instantiate with download=True to download it')
      images = numpy.full((43,100,100,3),255,dtype=numpy.uint8)
      with zipfile.ZipFile(pasrc) as z:
        for i in range(43):
          im = numpy.array(PIL.Image.open(z.open('resources/signs/%d.jpg'%i)))
          offset = (100-im.shape[0])//2
          images[i,offset:offset+im.shape[0]] = im
      self.cat_images = numpy.array(images)

    def load(self, verbose=True):
      ptraining   = os.path.join(self.root, self.raw_folder, os.path.basename(URL_TRAINING))
      ptest       = os.path.join(self.root, self.raw_folder, os.path.basename(URL_TEST))
      ptestlabels = os.path.join(self.root, self.raw_folder, os.path.basename(URL_TEST_LABELS))
      self.images = []
      self.labels = []
      self.batches = []
      if verbose: print ("Extracting training images")
      with zipfile.ZipFile(ptraining) as z:
        names = [n for n in z.namelist() if n.endswith('.ppm')]
        num = len(names)
        for i,n in enumerate(names):
          im = PIL.Image.open(z.open(n))
          if im.size[0]>=self.min_size[0] and im.size[1]>=self.min_size[1]:
            if self.target_size is not None:
              im = im.resize(self.target_size)
            self.images.append(numpy.array(im))
            self.labels.append(int(os.path.basename(os.path.dirname(n))))
            self.batches.append(int(os.path.basename(n).split('_')[0]))
      if self.target_size is not None: 
        self.images = numpy.array(self.images)
        imsize=3*numpy.prod(self.target_size)
        self.images = self.images-self.images.reshape((-1,imsize)).min(1).reshape((-1,1,1,1))
        self.images=(self.images*(255/self.images.reshape((-1,imsize)).max(1).reshape((-1,1,1,1)))).astype(numpy.uint8)
      self.labels = numpy.array(self.labels)
      self.batches = numpy.array(self.batches)
    
      if verbose: print ("Extracting test images")
      with zipfile.ZipFile(ptestlabels) as z:
        test_metadata = pandas.read_csv(z.open('GT-final_test.csv'), sep=';')
      test_metadata = test_metadata[(test_metadata["Width"]>=self.min_size[0])&(test_metadata["Height"]>=self.min_size[1])]

      num = len(test_metadata)  
      self.test_images = []
      self.test_labels = []
      with zipfile.ZipFile(ptest) as z:
        for i,(n,l) in enumerate(zip(test_metadata["Filename"],test_metadata["ClassId"])):
          im = PIL.Image.open(z.open('GTSRB/Final_Test/Images/%s'%n))
          if self.target_size is not None:
            im = im.resize(self.target_size)
          self.test_images.append(numpy.array(im))
          self.test_labels.append(l)
      if self.target_size is not None: 
        self.test_images = numpy.array(self.test_images)
        self.test_images = self.test_images-self.test_images.reshape((-1,imsize)).min(1).reshape((-1,1,1,1))
        self.test_images=(self.test_images*(255/self.test_images.reshape((-1,imsize)).max(1).reshape((-1,1,1,1)))).astype(numpy.uint8)
      self.test_labels = numpy.array(self.test_labels)
      if verbose: print ("Done loading")

      if self.split_validation>0.0:
        # Das ist etwas elaboriert, weil die Bilder im GTSRB nicht unabhängig sind, sondern jeweils in Episoden kommen
        # Der Validierungsdatensatz muss aber unabhängig sein(!)
        numclasses = self.labels.max()+1
        # Berechne das Maximum der Batches pro Klasse 
        maxbatchperclass = ((numpy.arange(numclasses,dtype=int)[numpy.newaxis]==self.labels[:,numpy.newaxis])*self.batches[:,numpy.newaxis]).max(0)
        # Berechne den letzten Trainingsindex pro Klasse
        lasttrainperclass = maxbatchperclass-numpy.fmax(numpy.array(maxbatchperclass*self.split_validation,dtype=int),1)
        # Test-Daten und Validierungsdaten als Bit-Masks
        in_train = (((numpy.arange(numclasses,dtype=int)[numpy.newaxis]==self.labels[:,numpy.newaxis])*self.batches[:,numpy.newaxis])<=lasttrainperclass).min(1)
        in_val  = ~in_train

        self.val_images = self.images[in_val]
        self.val_labels = self.labels[in_val]
        self.images = self.images[in_train]
        self.labels = self.labels[in_train]


        
    def download(self):
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            #with open(file_path.replace('.gz', ''), 'wb') as out_f, \
            #        gzip.GzipFile(file_path) as zip_f:
            #    out_f.write(zip_f.read())
            #os.unlink(file_path)

        # process and save as torch files
        #print('Processing...')

        #training_set = (
        #    read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
        #    read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        #)
        #test_set = (
        #    read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
        #    read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        #)
        #with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
        #    torch.save(training_set, f)
        #with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
        #    torch.save(test_set, f)

        #print('Done!')


categories = ["20","30","50","60","70",
              "80","Ende: 80","100","120","Überholverbot",
              "Überholverbot LKW", "Vorfahrt","Vorfahrtsstraße","Vorfahrt gewähren","Halt. Vorfahrt gewähren",
              "Verbot für Fahrzeuge aller Art","Verbot für LKW","Verbot der Einfahrt","Gefahrenstelle","Kurve (L)",
              "Kurve (R)","Doppelkurve (LR)","Unebene Fahrbahn","Schleudergefahr", "Einseitig verengte Fahrbahn (R)", 
              "Arbeitsstelle","Lichtzeichenanlage","Fußgänger","Kinder","Radverkehr",
              "Eisglätte","Wildwechsel","Ende Beschränkungen","Gebot: Rechts", "Gebot: Links", 
              "Gebot: Geradeaus","Gebot: Geradeaus oder rechts", "Gebot: Geradeaus oder links","Gebot Rechts vorbei","Gebot: Links vorbei", 
              "Gebot: Kreisverkehr", "Ende Überholverbot", "Ende Überholverbot LKW"]
          

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
  
