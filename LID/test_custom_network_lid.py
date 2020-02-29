#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 16:36:57 2020

@author: marius
"""

from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from util import train_lr

import tensorflow as tf
from tensorflow import keras

from keras.utils import to_categorical
from keras.datasets import cifar10

from matplotlib import pyplot as plt
plt.ioff()

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, average_precision_score
from sklearn.preprocessing import MinMaxScaler

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

# Pick the best values and test with them
winner_k = 10
winner_noise_scale = 0.0504 * np.sqrt(boxmax - boxmin)

# Reload and retrain the logistic classifier
# Load training data
contents = hdf5storage.loadmat('results/train_characteristics_k%d_sigma%.6f.mat' % (winner_k, winner_noise_scale))
train_characteristics = contents['train_characteristics']
train_labels = contents['train_labels']
shuffled_clean = contents['shuffled_clean']
shuffled_noisy = contents['shuffled_noisy']
shuffled_adv = contents['shuffled_adv']
# Train a scaler
scaler = MinMaxScaler().fit(train_characteristics)
train_characteristics = scaler.transform(train_characteristics)
train_labels          = train_labels[:,0]

# Train logistic regression
lr = train_lr(train_characteristics, train_labels)

# Load characteristics of clean test data
contents = hdf5storage.loadmat('results/test_clean_characteristics_k%d.mat' % winner_k)
test_clean_characteristics = contents['test_clean_characteristics']
# Run through scaler
test_clean_characteristics = scaler.transform(test_clean_characteristics)

# Test everything
test_strength_range = [8, 16, 64, 255]
test_success_rate = []
test_precision  = []
test_thresholds = []
test_auc = []
test_tpr = []
test_fpr = []

# For each test PGD strength
for test_strength in test_strength_range:
    # Load characteristics of attacked test data
    attack_test = 'PGD'
    contents    = hdf5storage.loadmat('results/test_adv_characteristics_k%d_%s_strength%d.mat' % (
            winner_k, attack_test, test_strength))
    test_adv_characteristics = contents['test_adv_characteristics']
    # Attack success rate
    test_success_rate.append(test_adv_characteristics.shape[0] / (256 * 128)) # This is how many batches/size we generated
    
    # Transform with scaler
    test_adv_characteristics = scaler.transform(test_adv_characteristics)
    
    # Evaluate together with clean data and get AUC
    merged_data = np.concatenate((test_clean_characteristics, test_adv_characteristics), axis=0)
    merged_labels = np.concatenate((np.zeros((test_clean_characteristics.shape[0],)),
                                    np.ones((test_adv_characteristics.shape[0],))), axis=0)
    # Predict
    y_test_pred = lr.predict_proba(merged_data)
    fpr, tpr, thresholds = roc_curve(merged_labels, y_test_pred[:, 1])
    avg_precision_score = average_precision_score(merged_labels, y_test_pred[:, 1], pos_label=0)
    area = auc(fpr, tpr)
    
    # Save
    test_auc.append(area)
    test_tpr.append(tpr)
    test_fpr.append(fpr)
    test_precision.append(avg_precision_score)
    test_thresholds.append(thresholds)