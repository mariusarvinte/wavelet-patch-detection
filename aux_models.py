#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 18:58:02 2020

@author: marius
"""

import tensorflow as tf


# Assignment rather than import because direct import from within Keras
# doesn't work in tf 1.8
Sequential = tf.keras.models.Sequential
Conv2D = tf.keras.layers.Conv2D
Input  = tf.keras.layers.Input
Dense = tf.keras.layers.Dense
Activation = tf.keras.layers.Activation
Flatten = tf.keras.layers.Flatten
Lambda = tf.keras.layers.Lambda
KerasModel = tf.keras.models.Model
MaxPooling2D = tf.keras.layers.MaxPooling2D
Concatenate = tf.keras.layers.Concatenate

TFKerasModel = tf.keras.Model

def conv_2d(filters, kernel_shape, strides, padding, input_shape=None):
  """
  Defines the right convolutional layer according to the
  version of Keras that is installed.
  :param filters: (required integer) the dimensionality of the output
                  space (i.e. the number output of filters in the
                  convolution)
  :param kernel_shape: (required tuple or list of 2 integers) specifies
                       the kernel shape of the convolution
  :param strides: (required tuple or list of 2 integers) specifies
                       the strides of the convolution along the width and
                       height.
  :param padding: (required string) can be either 'valid' (no padding around
                  input or feature map) or 'same' (pad to ensure that the
                  output feature map size is identical to the layer input)
  :param input_shape: (optional) give input shape if this is the first
                      layer of the model
  :return: the Keras layer
  """
  if input_shape is not None:
    return Conv2D(filters=filters, kernel_size=kernel_shape,
                  strides=strides, padding=padding,
                  input_shape=input_shape)
  else:
    return Conv2D(filters=filters, kernel_size=kernel_shape,
                  strides=strides, padding=padding)

def vgg19_model(logits=False, input_ph=None, img_rows=28, img_cols=28,
              channels=1, nb_classes=10):
      """
      Defines a CNN model using Keras sequential model
      :param logits: If set to False, returns a Keras model, otherwise will also
                      return logits tensor
      :param input_ph: The TensorFlow tensor for the input
                      (needed if returning logits)
                      ("ph" stands for placeholder but it need not actually be a
                      placeholder)
      :param img_rows: number of row in the image
      :param img_cols: number of columns in the image
      :param channels: number of color channels (e.g., 1 for MNIST)
      :param nb_filters: number of convolutional filters per layer
      :param nb_classes: the number of output classes
      :return:
      """
      model = Sequential()
    
      # Define the layers successively (convolution layers are version dependent)
      if tf.keras.backend.image_data_format() == 'channels_first':
        input_shape = (channels, img_rows, img_cols)
      else:
        assert tf.keras.backend.image_data_format() == 'channels_last'
        input_shape = (img_rows, img_cols, channels)
    
      
      layers = [conv_2d(64, (3, 3), (1, 1), 'same',
                        input_shape=input_shape),
      Activation('relu'),
      conv_2d(64, (3, 3), (1, 1), 'same'),
      Activation('relu'),
      MaxPooling2D((2, 2), strides=(2, 2)),
      conv_2d(128, (3, 3), (1, 1), 'same'),
      Activation('relu'),
      conv_2d(128, (3, 3), (1, 1), 'same'),
      Activation('relu'),
      MaxPooling2D((2, 2), strides=(2, 2)),
      conv_2d(256, (3, 3), (1, 1), 'same'),
      Activation('relu'),
      conv_2d(256, (3, 3), (1, 1), 'same'),
      Activation('relu'),
      conv_2d(256, (3, 3), (1, 1), 'same'),
      Activation('relu'),
      conv_2d(256, (3, 3), (1, 1), 'same'),
      Activation('relu'),
      MaxPooling2D((2, 2), strides=(2, 2)),
      conv_2d(512, (3, 3), (1, 1), 'same'),
      Activation('relu'),
      conv_2d(512, (3, 3), (1, 1), 'same'),
      Activation('relu'),
      conv_2d(512, (3, 3), (1, 1), 'same'),
      Activation('relu'),
      conv_2d(512, (3, 3), (1, 1), 'same'),
      Activation('relu'),
      MaxPooling2D((2, 2), strides=(2, 2)),
      conv_2d(512, (3, 3), (1, 1), 'same'),
      Activation('relu'),
      conv_2d(512, (3, 3), (1, 1), 'same'),
      Activation('relu'),
      conv_2d(512, (3, 3), (1, 1), 'same'),
      Activation('relu'),
      conv_2d(512, (3, 3), (1, 1), 'same'),
      Activation('relu'),
      MaxPooling2D((2, 2), strides=(2, 2)),
      Flatten(),
      Dense(256),
      Activation('relu'),
      Dense(nb_classes)]
    
      for layer in layers:
        model.add(layer)
    
      if logits:
        logits_tensor = model(input_ph)
    
      model.add(Activation('softmax'))
    
      if logits:
        return model, logits_tensor
      else:
        return model

# Residual-based anomaly detector
# Takes as input an image and returns a 2D one-hot vector
def detector_model(logits=False, input_ph=None, img_rows=28, img_cols=28,
              channels=1, nb_classes=10):
    # Limited to Sequential(), can be turned into Model()
    model = Sequential()
    
    # Define the layers successively (convolution layers are version dependent)
    if tf.keras.backend.image_data_format() == 'channels_first':
        input_shape = (channels, img_rows, img_cols)
    else:
        assert tf.keras.backend.image_data_format() == 'channels_last'
        input_shape = (img_rows, img_cols, channels)

    # Body
    layers = [conv_2d(32, (3, 3), (1, 1), 'same',
                      input_shape=input_shape),
    Activation('relu'),
    MaxPooling2D((2, 2), strides=(2, 2)),
    conv_2d(64, (3, 3), (1, 1), 'same'),
    Activation('relu'),
    MaxPooling2D((2, 2), strides=(2, 2)),
    Flatten(),
    Dense(64),
    Activation('relu'),
    Dense(nb_classes)]
    
    for layer in layers:
       model.add(layer)

    if logits:
        logits_tensor = model(input_ph)

    model.add(Activation('softmax'))

    if logits:
        return model, logits_tensor
    else:
        return model

# Copy-cat anomaly detector
def copycat_model(logits=False, input_ph=None, img_rows=28, img_cols=28,
              channels=1, nb_classes=10):
    # Limited to Sequential(), can be turned into Model()
    # But will require mandatory input tensor, at least in TF1.X
    model = Sequential()
    
    # Define the layers successively (convolution layers are version dependent)
    if tf.keras.backend.image_data_format() == 'channels_first':
        input_shape = (channels, img_rows, img_cols)
    else:
        assert tf.keras.backend.image_data_format() == 'channels_last'
        input_shape = (img_rows, img_cols, channels)

    # Body
    layers = [conv_2d(32, (3, 3), (1, 1), 'same',
                      input_shape=input_shape),
    Activation('relu'),
    conv_2d(32, (3, 3), (1, 1), 'same'),
    Activation('relu'),
    MaxPooling2D((2, 2), strides=(2, 2)),
    conv_2d(64, (3, 3), (1, 1), 'same'),
    Activation('relu'),
    conv_2d(64, (3, 3), (1, 1), 'same'),
    Activation('relu'),
    MaxPooling2D((2, 2), strides=(2, 2)),
    Flatten(),
    Dense(64),
    Activation('relu'),
    Dense(nb_classes)]
    
    for layer in layers:
       model.add(layer)

    if logits:
        logits_tensor = model(input_ph)

    model.add(Activation('softmax'))

    if logits:
        return model, logits_tensor
    else:
        return model