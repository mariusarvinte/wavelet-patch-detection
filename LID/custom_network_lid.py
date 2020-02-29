#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 16:36:57 2020

@author: marius
"""

from __future__ import absolute_import
from __future__ import print_function

import sys
sys.path.append('../')

import numpy as np

from extract_characteristics import get_lid
from util import (get_model, get_data, get_noisy_samples, get_mc_predictions,
                      get_deep_representations, score_samples, normalize,
                      get_lids_random_batch, get_kmeans_random_batch)

import tensorflow as tf
from tensorflow import keras

from keras.utils import to_categorical
from keras.datasets import cifar10

from matplotlib import pyplot as plt
plt.ioff()

from sklearn.model_selection import train_test_split

from aux_models import vgg19_model

import hdf5storage

# Sanity
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(1)
sess = tf.Session(config=config).__enter__()
tf.keras.backend.set_learning_phase(0)
# Set TF random seed to improve reproducibility
tf.set_random_seed(1234)
# Create TF session and set as Keras backend session
keras.backend.set_session(sess)

# Dataset parameters
num_classes = 10
output_dim = num_classes
input_h, input_w, input_c = 32, 32, 3
input_shape = (input_h, input_w, input_c)
class_name = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck',
}

# Load CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Normalize
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')
# Further training/validation split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1,
                                                  random_state=2019)
# Save clean images (normalized)
x_clean_val = np.copy(x_val) / 255.
# Derive statistics based on training data
train_mean = np.mean(x_train, axis=(0,1,2,3))
train_std  = np.std(x_train, axis=(0,1,2,3))
# Center with stabilizer
x_train = (x_train - train_mean) / (train_std + 1e-7)
x_val   = (x_val - train_mean) / (train_std + 1e-7)
x_test  = (x_test - train_mean) / (train_std + 1e-7)
# Convert to one-hot
y_train = to_categorical(y_train, num_classes)
y_val   = to_categorical(y_val, num_classes)
y_test  = to_categorical(y_test, num_classes)
# Boxes
boxmin = (0 - train_mean) / (train_std + 1e-7)
boxmax = (255 - train_mean) / (train_std + 1e-7)

# Weights
weight_path  = '../models/weights_clean_best.h5'
# Boxes
boxmin = (0. - train_mean) / (train_std + 1e-7)
boxmax = (255. - train_mean) / (train_std + 1e-7) + 1e-6
# Obtain Image Parameters
img_rows, img_cols, nchannels = x_train.shape[1:4]
nb_classes = y_train.shape[1]

# Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                      nchannels))
y = tf.placeholder(tf.float32, shape=(None, nb_classes))
y_target = tf.placeholder(tf.float32, shape=(None, nb_classes))

# Define TF model graph
model = vgg19_model(img_rows=img_rows, img_cols=img_cols,
                  channels=nchannels,
                  nb_classes=nb_classes)
# Load weights
model.load_weights(weight_path)

# Hyperparameters
#k_nearest_range = [10, 20, 40]
k_nearest_range = [10]
batch_size      = 200
# With the values used in the original source code, but modified for our scale
#noise_scale_range = np.asarray([0.0504/10, 0.0504, 0.0504*10]) * np.sqrt(boxmax - boxmin)
noise_scale_range = np.asarray([0.0504]) * np.sqrt(boxmax - boxmin)
# Load training data - concatenated
attack_train = 'pgd'
train_strength_range = [8, 16]
train_adv   = np.empty((0, input_w, input_h, input_c))
train_clean = np.empty((0, input_w, input_h, input_c)) 
for train_strength in train_strength_range:
    attack_step = train_strength / 10
    contents    = hdf5storage.loadmat('../cifar10_blackbox_samples/%s_train_strength%d_step%.2f.mat' % (attack_train, train_strength, attack_step))
    train_adv   = np.append(train_adv, contents['x_adv_val'], axis=0)
    # Clean counterparts
    train_clean = np.append(train_clean, contents['x_clean_val'], axis=0)

for noise_scale in noise_scale_range:
    for k_nearest in k_nearest_range:
        # Training collections
        train_characteristics = np.empty((0, 41)) # Hardcoded for VGG19 to use all layer representations
        train_labels = np.empty((0, 1))
        
        # Sample a number of meta-batches of ~1000 points each and extract their characteristics to avoid duplicates
        meta_batches = 10
        meta_batch_size = 3000
        shuffled_clean = np.empty((0, input_w, input_h, input_c))
        shuffled_noisy = np.empty((0, input_w, input_h, input_c))
        shuffled_adv = np.empty((0, input_w, input_h, input_c))
        
        for meta_idx in range(meta_batches):
            # Permute
            permute_idx = np.random.permutation(np.arange(train_clean.shape[0]))
            # Downselect
            local_clean = train_clean[permute_idx[:meta_batch_size]]
            local_adv   = train_adv[permute_idx[:meta_batch_size]]
            # Further filter duplicates
            _, unique_idx = np.unique(local_clean, axis=0, return_index=True)
            local_clean = local_clean[unique_idx]
            local_noisy = np.clip(local_clean + np.random.normal(loc=0., scale=noise_scale,
                                                     size=local_clean.shape), boxmin, boxmax)
            local_adv   = local_adv[unique_idx]
            # Print and get their characteristics and labels
            print('Current batch has %d points.' % local_clean.shape[0])
            local_characteristics, local_labels = get_lid(model, local_clean, local_noisy, local_adv,
                                          k_nearest, batch_size, 'cifar')
            # Append
            train_characteristics = np.append(train_characteristics, local_characteristics, axis=0)
            train_labels          = np.append(train_labels, local_labels, axis=0)
            # And auxiliaries for sanity checking
            shuffled_clean = np.append(shuffled_clean, local_clean, axis=0)
            shuffled_noisy = np.append(shuffled_noisy, local_noisy, axis=0)
            shuffled_adv   = np.append(shuffled_adv, local_adv, axis=0)
        
        # Save to .mat file without scaling
        hdf5storage.savemat('results/train_characteristics_k%d_sigma%.6f.mat' % (k_nearest, noise_scale),
                            {'train_characteristics': train_characteristics,
                             'train_labels': train_labels,
                             'shuffled_clean': shuffled_clean,
                             'shuffled_noisy': shuffled_noisy,
                             'shuffled_adv': shuffled_adv})