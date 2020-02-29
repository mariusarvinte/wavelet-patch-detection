#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:39:15 2020

@author: marius
"""


from keras.layers import Input, Dense, Conv2D
from keras.layers import Activation
from keras.layers import Lambda, Concatenate
from keras.layers import Flatten, Dropout
from keras.layers import MaxPool2D, AvgPool2D
from keras.layers import Permute, Reshape, Multiply
from keras.layers import BatchNormalization

from keras.models import Model
from keras.initializers import glorot_uniform, he_uniform
from keras.regularizers import l2
from keras import backend as K

import cv2
import numpy as np
import tensorflow as tf

# Image residual classifier with a single logit
def BinaryResidueClassifier(input_w, input_h, input_c,
                      weight_reg):
    
    # Weight regularizers
    l2_reg = l2(weight_reg)
    
    # Input
    input_img = Input(shape=(input_h, input_w, input_c))
    
    # Block 1
    x = Conv2D(32, (3, 3),
               activation='relu',
               padding='same',
               name='block1_conv1',
               kernel_regularizer=l2_reg)(input_img)
    x = MaxPool2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    
    # Block 2
    x = Conv2D(64, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv2',
               kernel_regularizer=l2_reg)(x)
    x = MaxPool2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    
    # Flatten
    encoded = Flatten()(x)
    # FC
    encoded = Dense(64, activation='relu', kernel_regularizer=l2_reg)(encoded)
    encoded = Dropout(rate=0.3)(encoded)
    
    # Output logits
    encoded = Dense(1, activation='linear', kernel_regularizer=l2_reg)(encoded)
    exposed_logits = encoded
    
    # Output probabilities
    encoded = Activation(activation='softmax')(encoded)
    
    # Model
    classifier = Model(input_img, encoded)
    logit_exposer = Model(input_img, exposed_logits)
    classifier.summary()
    
    return classifier, logit_exposer

# Image residual and logit joint classifier
def ResidueLogitClassifier(input_w, input_h, input_c,
                      weight_reg):
    
    # Weight regularizers
    l2_reg = l2(weight_reg)
    
    # Input
    input_img    = Input(shape=(input_h, input_w, input_c))
    input_logits = Input(shape=(10,)) 
    
    # Block 1
    x = Conv2D(32, (3, 3),
               activation='relu',
               padding='same',
               name='block1_conv1',
               kernel_regularizer=l2_reg)(input_img)
    x = MaxPool2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    
    # Block 2
    x = Conv2D(64, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv2',
               kernel_regularizer=l2_reg)(x)
    x = MaxPool2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    
    # Flatten
    encoded = Flatten()(x)
    # FC
    encoded = Dense(64, activation='relu', kernel_regularizer=l2_reg)(encoded)
    encoded = Dropout(rate=0.3)(encoded)
    
    # Merge with logits
    encoded = Concatenate(axis=-1)([encoded, input_logits])
    
    # Output (logistic)
    encoded = Dense(2, activation='softmax', kernel_regularizer=l2_reg)(encoded)
    
    # Model
    classifier = Model([input_img, input_logits], encoded)
    classifier.summary()
    
    return classifier

# Image residual classifier
def SmallResidueClassifier(input_w, input_h, input_c,
                      weight_reg):
    
    # Weight regularizers
    l2_reg = l2(weight_reg)
    
    # Input
    input_img = Input(shape=(input_h, input_w, input_c))
    
    # Filter image
#    x = AvgPool2D(pool_size=(2, 2))(input_img)
    
    # Block 1
    x = Conv2D(32, (3, 3),
               activation='relu',
               padding='same',
               name='block1_conv1',
               kernel_regularizer=l2_reg)(input_img)
    x = MaxPool2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    
    # Flatten
    encoded = Flatten()(x)
    # FC
    encoded = Dense(32, activation='relu', kernel_regularizer=l2_reg)(encoded)
    encoded = Dropout(rate=0.3)(encoded)
    
    # Output logits
    encoded = Dense(2, activation='linear', kernel_regularizer=l2_reg)(encoded)
    exposed_logits = encoded
    
    # Output probabilities
    encoded = Activation('softmax')(encoded)
    
    # Model
    classifier = Model(input_img, encoded)
    logit_exposer = Model(input_img, exposed_logits)
    classifier.summary()
    
    return classifier, logit_exposer

# Image residual classifier
def ResidueClassifier(input_w, input_h, input_c,
                      weight_reg):
    
    # Weight regularizers
    l2_reg = l2(weight_reg)
    
    # Input
    input_img = Input(shape=(input_h, input_w, input_c))
    
    # Block 1
    x = Conv2D(32, (3, 3),
               activation='relu',
               padding='same',
               name='block1_conv1',
               kernel_regularizer=l2_reg)(input_img)
    x = MaxPool2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    
    # Block 2
    x = Conv2D(64, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv2',
               kernel_regularizer=l2_reg)(x)
    x = MaxPool2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    
    # Flatten
    encoded = Flatten()(x)
    # FC
    encoded = Dense(64, activation='relu', kernel_regularizer=l2_reg)(encoded)
    encoded = Dropout(rate=0.3)(encoded)
    
    # Output logits
    encoded = Dense(2, activation='linear', kernel_regularizer=l2_reg)(encoded)
    exposed_logits = encoded
    
    # Output probabilities
    encoded = Activation('softmax')(encoded)
    
    # Model
    classifier = Model(input_img, encoded)
    logit_exposer = Model(input_img, exposed_logits)
    classifier.summary()
    
    return classifier, logit_exposer

# Scratch-trained classifier from hard labels, to be used as distilled gradient
def UnknownClassifier(input_w, input_h, input_c,
                      weight_reg):
    # Weight regularizers
    l2_reg = l2(weight_reg)
    
    # Input
    input_img = Input(shape=(input_h, input_w, input_c))
    
    # Block 1
    x = Conv2D(32, (3, 3),
               activation='relu',
               padding='same',
               name='block1_conv1',
               kernel_regularizer=l2_reg)(input_img)
    x = Conv2D(32, (3, 3),
               activation='relu',
               padding='same',
               name='block1_conv2',
               kernel_regularizer=l2_reg)(x)
    x = MaxPool2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    
    # Block 2
    x = Conv2D(64, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv2',
               kernel_regularizer=l2_reg)(x)
    x = Conv2D(64, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv3',
               kernel_regularizer=l2_reg)(x)
    x = MaxPool2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    
    # Flatten
    encoded = Flatten()(x)
    # FC
    encoded = Dense(64, activation='relu', kernel_regularizer=l2_reg)(encoded)
    encoded = Dropout(rate=0.3)(encoded)
    
    # Output
    encoded = Dense(2, activation='softmax', kernel_regularizer=l2_reg)(encoded)
    
    # Model
    classifier = Model(input_img, encoded)
    classifier.summary()
    
    return classifier