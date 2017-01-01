#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2017 Thomas Viehmann http://lernapparat.de/
# Licensed under the MIT license

import keras
import GTSRB_utils
from sklearn.metrics import confusion_matrix

images, labels, val_images, val_labels, test_images, test_labels = GTSRB_utils.get_GTSRB_data()
features, val_features, test_features = GTSRB_utils.get_resnet50_preprocessed(images, val_images, test_images)

GTSRB_utils.plot_sample(images)

m = GTSRB_utils.get_resnet50_top(labels.shape[1])
m.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history = m.fit(features,labels, validation_data=(val_features,val_labels), nb_epoch=50)
GTSRB_utils.plot_history(history)

test_predictions = m.predict(test_features)

# die für am wahrscheinlichsten gehaltenen Klassen
test_predicted_classes = test_predictions.argmax(1)
test_label_classes     = test_labels.argmax(1)

cm = confusion_matrix(test_label_classes,test_predicted_classes)

print ("Test accuracy:",(test_label_classes==test_predicted_classes).mean())
print (GTSRB_utils.maximal_confusion_str(cm))
GTSRB_utils.plot_confusion_matrix(cm)
GTSRB_utils.plot_errors(test_predictions, test_labels, test_images)
