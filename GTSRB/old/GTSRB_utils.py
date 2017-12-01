#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Code (c) 2017 by Thomas Viehmann <tv.code@beamnet.de>
# licensed under the MIT License

import numpy
import os
import h5py
import keras
import keras.applications
import GTSRB_data
from matplotlib import pyplot
import itertools

NEW_CACHE_FN = "gtsrb_data_cache.h5"
resnet50_cached = None


def delete_cache(CACHE_FN, name):
  if os.path.exists(CACHE_FN):
    with h5py.File(CACHE_FN, "r+") as cf:
      del cf[name]
  
def compute_or_cache(CACHE_FN, name):
  def decorate(fun):
    def g(*v,**k):
      if not os.path.exists(CACHE_FN):
        h5py.File(CACHE_FN,"w").close() # not thread safe...
      with h5py.File(CACHE_FN, "r+") as cf:
        if name in cf:
          res = []
          numres = cf[name].attrs['numres']
          if numres > 0:
            for i in range(numres):
              res.append(numpy.array(cf['%s/r%d'%(name,i)]))
            return res
          else:
            return (numpy.array(cf['%s/r'%(name,)]))
        else:
          res = fun(*v, **k)
          dset = cf.create_group(name)
          if isinstance(res,numpy.ndarray):
            dset.attrs['numres'] = 0
            dset['r'] = res
          else:
            dset.attrs['numres'] = len(res)
            for i,r in enumerate(res):
              dset['r%d'%(i,)] = r
        return res
    return g
  return decorate

class FixedRandomContext:
  """
  Context manager to use fixed random numbers locally
  without disturbing the randomness of everything else.
  
  Use as:
  with FixedRandomContext():
    something here
  """
  def __init__(self,seed=12345678):
    self.seed = seed
    self.state = None
  def __enter__(self):
    self.state = numpy.random.get_state()
    numpy.random.seed(self.seed)
  def __exit__(self, ex_type, ex_val, tb):
    numpy.random.set_state(self.state)

def get_GTSRB_data(split_validation=0.1):
  """
  Lädt die GTSRB-Daten.
  
  Um die Bilder (im Original nach Klassen sortiert) zu mischen, wird das
  Trainings-Set mit einem festen Random-Seed permutiert. 
  
  Eingaben:
  split_validation -- Relativer Anteil des Validierungs-Datensatzes am Trainings-Datensatz (Standard: 0.1)
                      Dies wird approximativ erreicht, weil die Bilder im GTSRB nicht unabhängig sind, sondern
                      in Episoden kommen und für die Unabhängigkeit entlang ganzer Episoden geteilt wird.
                      (Der relative Anteil wird auf die Anzahl Episoden genommen.)
  Rückgabewerte:
  images, labels, test_images, test_labels falls keine split_validation nicht positiv
  oder
  images, labels, val_images, val_labels, test_images, test_labels falls keine split_validation positiv
  """
  (images,labels, batches),(test_images, test_labels) = GTSRB_data.load_data(verbose=True, return_batches=True)

  if split_validation>0.0:
    # Das ist etwas elaboriert, weil die Bilder im GTSRB nicht unabhängig sind, sondern jeweils in Episoden kommen
    # Der Validierungsdatensatz muss aber unabhängig sein(!)
    numclasses = labels.max()+1
    # Berechne das Maximum der Batches pro Klasse 
    maxbatchperclass = ((numpy.arange(numclasses,dtype=int)[numpy.newaxis]==labels[:,numpy.newaxis])*batches[:,numpy.newaxis]).max(0)
    # Berechne den letzten Trainingsindex pro Klasse
    lasttrainperclass = maxbatchperclass-numpy.fmax(numpy.array(maxbatchperclass*split_validation,dtype=int),1)
    # Test-Daten und Validierungsdaten als Bit-Masks
    in_train = (((numpy.arange(numclasses,dtype=int)[numpy.newaxis]==labels[:,numpy.newaxis])*batches[:,numpy.newaxis])<=lasttrainperclass).min(1)
    in_val  = ~in_train

    val_images = images[in_val]
    val_labels = keras.utils.np_utils.to_categorical(labels[in_val])
    images = images[in_train]
    labels = labels[in_train]

  with FixedRandomContext():
     idx = numpy.random.permutation(len(images))       
  images = images[idx]
  labels = keras.utils.np_utils.to_categorical(labels[idx])

  test_labels = keras.utils.np_utils.to_categorical(test_labels)
 
  if split_validation>0.0:
    return images, labels, val_images, val_labels, test_images, test_labels
  return images, labels, test_images, test_labels

def get_resnet50_top(output_dimension):
  """
  Gibt eine zu den ResNet50-Features passende Klassifikationsschicht zurück.
  
  Wie hier mit einer Schicht an den Feature-Teil anzuschließen entspricht dem ResNet50-Modell.
  In anderen Modellen werden oft 1-2 Schichten zwischen Features und Ausgabeschicht eingebaut.
  Eingaben:
  output_dimension -- ist die Anzahl der Klassen.
  Rückgabe:
  ein Keras-Modell mit einer Schicht.
  """
  m = keras.models.Sequential(
       [keras.layers.Dense(output_dimension,activation='softmax', input_shape=(2048,), name='predictions')])
  return m

def get_resnet50_two_layer_top(output_dimension):
  """
  Gibt ein zu den ResNet50-Features passendes 2-Schicht Klassifikationsmodell zurück.
  
  Das ResNet50-Modell verwendet für den Klassifikations-Teil nur eine Schicht.
  In anderen Modellen werden oft 1-2 Schichten zwischen Features und Ausgabeschicht eingebaut.
  Eingaben:
  output_dimension -- ist die Anzahl der Klassen.
  Rückgabe:
  ein Keras-Modell mit zwei Schichten.
  """
  m = keras.models.Sequential(
       [keras.layers.Dense(512,activation='relu', input_shape=(2048,), name='fc1'),
        keras.layers.Dense(output_dimension,activation='softmax', name='predictions')])
  return m

def preprocess_images(images):
  """Gibt eine Kopie der übergebenen Bilder, die direkt in ImageNet-Modelle wie ResNet und VGG gesteckt werden kann, zurück."""
  return keras.applications.imagenet_utils.preprocess_input(numpy.array(images,dtype=numpy.float32,copy=True))

@compute_or_cache(NEW_CACHE_FN, "resnet50_preprocessed")
def get_resnet50_preprocessed(images, val_images, test_images):
  """
  Berechnet zu images, val_images und test_images die ResNet50-Features mit den unteren 49 Schichten.
  
  *Obacht*: die Funktion nimmt zwar Argumente, aber
  
  - wenn kein Cache vorhanden ist, werden die übergebenen Argumente in der Berechnung benutzt.
    Das Ergebnis wird im Cache gespeichert.

  - wenn ein Cache vorhanden ist, wird dieser zurückgegeben, egal welche Argumente übergeben werden.
  
  """
  global resnet50_cached
  if resnet50_cached is None:
    resnet50_cached = keras.applications.ResNet50()
  m_pre = keras.models.Model(input=resnet50_cached.input, output=resnet50_cached.layers[-1].input)
  print ("Precalculating last layer from ResNet50.")
  print ("for training inputs ", end="")
  pre_training = m_pre.predict(preprocess_images(images), verbose=1)
  print ("for validation inputs ", end="")
  pre_val = m_pre.predict(preprocess_images(val_images), verbose=1)
  print ("for test inputs ", end="")
  pre_test = m_pre.predict(preprocess_images(test_images), verbose=1)
  return pre_training, pre_val, pre_test

def predict_resnet50(images):
  """Wertet das ResNet50 in der Originalkonfiguration aus."""
  global resnet50_cached
  if resnet50_cached is None:
    resnet50_cached = keras.applications.ResNet50()
  return resnet50_cached.predict(preprocess_images(images))

def decode_predictions(predictions, top=3):
  """Gibt die zu einem Array aus Wahrscheinlichkeitsvektoren gehörenden Image-Net Kategorien
  mit Wahrscheinlichkeiten zurück.
  
  Eingaben:
  predictions -- die vom Neuronalen-Netz zurückgegebenen Klassen-Wahrscheinlichkeiten
  top -- Anzahl der Vorhersagen, aus jedem Vektor berücksichtigt werden soll (Standardwert: 3)
  Rückgabe:
  Liste (Anzahl Bilder) von Listen (Anzahl top) von Vorhersagen, jeweils als (Klassenname, Wahrscheinlichkeit)
  """
  pred = keras.applications.imagenet_utils.decode_predictions(predictions,top=top)
  labels = [[l[1:] for l in labels] for labels in pred]
  return labels

def plot_confusion_matrix(cm, normalize=False, title='', cmap=pyplot.cm.Blues):
  #Adapted from a sklearn example.
  """
  Gibt die Konfusions-Matrix aus.
  
  Eingaben:
  cm -- Konfusionsmatrix
  normalize -- Gib relative Häufigkeiten statt Anzahlen aus (Standard: False)
  title -- Plot-Titel (Standard: keiner)
  cmap -- Matplotlib Colormap, z.B. aus pyplot.cm (Standard: Blautöne)
  """
  if title:
    title += ' '
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
    title += "(Normalized confusion matrix)"
  else:
    title += '(Confusion matrix, without normalization)'

  pyplot.figure(figsize=(15,15))
  ax = pyplot.subplot(1,1,1)
  classimgs = GTSRB_data.load_cat_info()
  for i,im in enumerate(classimgs):
    ax.imshow(im,extent=(i-0.5,i+0.5,i+0.5,i-0.5),zorder=10)
  cmnodiag = cm*(1-numpy.eye(cm.shape[0]))
  img = ax.imshow(cmnodiag, interpolation='nearest', cmap=cmap)
  ax.set_title(title)
  ax.set_ylabel('True')
  ax.set_xlabel('Predicted')
  pyplot.colorbar(img,ax=ax)
  
  cmmax = cmnodiag.max()
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if i!=j and cm[i,j]>0:
      ax.text(j, i, cm[i, j],
              horizontalalignment="center",
              verticalalignment="center",
              color="white" if cm[i, j] > cmmax/2 else "black",
              size='smaller')


def maximal_confusion(cm, num=10):
  """
  Gibt die häufigsten Verwechselungen aus.
  
  Eingaben:
  cm -- Konfusionsmatrix
  num -- Anzahl der Verwechselungspaare, die ausgegeben werden soll (Standard: 10)
  Rückgabe:
  Liste von Tripeln (wahre Klasse, vorhergesagte Klasse, Anzahl)
  """
  assert cm.shape[0]==cm.shape[1]
  cmnodiag = cm*(1-numpy.eye(cm.shape[0]))
  s = cmnodiag.flatten().argsort()[::-1]
  idx1 = numpy.arange(cm.shape[0]).repeat(cm.shape[0])[s]
  idx2 = numpy.tile(numpy.arange(cm.shape[0]),(cm.shape[0],))[s]
  confsorted = cm[idx1,idx2]
  conflist = [(a,b,c) for a,b,c in zip(idx1,idx2,confsorted) if c>0]
  conflist = conflist[:num]
  return conflist

def maximal_confusion_str(cm, num=10):
  """
  Gibt die häufigsten Verwechselungen im GTSRB als Text aus.
  
  Eingaben:
  cm  -- Konfusionsmatrix
  num -- Anzahl der Verwechselungspaare, die ausgegeben werden soll (Standard: 10)
  Rückgabe:
  String mit Zeilen der Form wahre Klasse -> vorhergesagte Klasse (Anzahl)
  """
  cats = numpy.array(GTSRB_data.categories) 
  return '\n'.join(['%s->%s (%d)'%(cats[a],cats[b],c) for a,b,c in maximal_confusion(cm, num=num)])

def plot_errors(test_predictions, test_labels, test_images, num=12, shuffle=True):
  """
  Gibt die  Verwechselungen im GTSRB mit wahrer und vorhergesagter Klasse als Bilder aus.
  
  Eingaben:
  test_predictions -- Vorhergesagte Klassen für den Test-Datensatz
  test_labels      -- Wahre Klassen für den Test-Datensatz
  test_images      -- Bilder für den Test-Datensatz
  num              -- Anzahl der Verwechselungspaare, die ausgegeben werden soll (Standard: 12)
  shuffle          -- Ziehe die Datensätze mit Fehlern zufällig (Standard: True)
  """  
  test_errors_images = test_images[test_predictions.argmax(1)!=test_labels.argmax(1)]
  test_errors_predictions = test_predictions[test_predictions.argmax(1)!=test_labels.argmax(1)]
  test_errors_labels = test_labels[test_predictions.argmax(1)!=test_labels.argmax(1)]
  if shuffle:
    idx = numpy.random.permutation(len(test_errors_images))
    test_errors_images = test_errors_images[idx[:num]]
    test_errors_predictions = test_errors_predictions[idx[:num]]
    test_errors_labels = test_errors_labels[idx[:num]]
  else:
    test_errors_images = test_errors_images[:num]
    test_errors_predictions = test_errors_predictions[:num]
    test_errors_labels = test_errors_labels[:num]
  pyplot.figure(figsize=(10,8))
  for i,(img,p,l) in enumerate(zip(test_errors_images,test_errors_predictions.argmax(1),test_errors_labels.argmax(1))): 
    ax = pyplot.subplot((num+3)//4,4,i+1)
    pyplot.title("%s\n->%s"%(GTSRB_data.categories[l],GTSRB_data.categories[p]))
    pyplot.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])

def plot_sample(images, num=12, shuffle=True):
  """
  Gibt die beispielhafte Bilder aus.
  
  Eingaben:
  images      -- Bilder
  num         -- Anzahl der Bilder, die ausgegeben werden soll (Standard: 12)
  shuffle     -- Ziehe die Bilder (Standard: True)
  """
  if shuffle:
    idx = numpy.random.permutation(len(images))
    images = images[idx[:num]]
  else:
    images = images[:num]
  pyplot.figure(figsize=(10,8))
  for i,img in enumerate(images): 
    ax = pyplot.subplot((num+3)//4,4,i+1)
    pyplot.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])

def plot_history(history, name='acc'):
  """
  Gibt die Entwicklung von Kenngrößen aus, falls vorhanden für Trainings- und Validierungsdaten.
  
  Eingaben:
  history     -- Entwicklungs-Log, wie es von der Modell-fit-Funktion zurückgegeben wird.
  name        -- Anzahl der Bilder, die ausgegeben werden soll, z.B. 'acc', 'loss' (Standard: 'acc')
  """
  epochnums = numpy.array(history.epoch)+1
  fig = pyplot.figure()
  ax = fig.add_subplot(1,1,1)
  ax.plot(epochnums,history.history[name],label='train '+name)
  if 'val_'+name in history.history:
    ax.plot(epochnums,history.history['val_'+name],label='val '+name)
  ax.legend(loc="lower right")
  ax.set_title(name)
  ax.set_xlabel('epoch')
