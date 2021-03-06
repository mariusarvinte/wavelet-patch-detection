#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 16:36:57 2020

@author: marius
"""

from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from util import (get_model, get_data, get_noisy_samples, get_mc_predictions,
                      get_deep_representations, score_samples, normalize,
                      get_lids_random_batch, get_kmeans_random_batch,
                      mle_batch, get_layer_wise_activations)
from util import train_lr

from tqdm import tqdm
from keras import backend as K
import tensorflow as tf
from tensorflow import keras

from keras.utils import to_categorical
from keras.datasets import cifar10

from matplotlib import pyplot as plt
plt.ioff()

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler

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

# Training and validation data
batch_size  = 256
num_batches = 16
# Find correctly classified data - these are implicitly generated by the attack script
y_hat = model.predict(x_val)
valid_idx = np.where(np.argmax(y_hat, axis=-1) == np.argmax(y_val, axis=-1))
x_valid = x_val[valid_idx]
y_valid = y_val[valid_idx]
# Targets
x_det_train = x_valid[:batch_size*num_batches]
y_det_train = y_valid[:batch_size*num_batches]
x_det_val   = x_valid[batch_size*num_batches:]
y_det_val   = y_valid[batch_size*num_batches:]

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

# Unique training data
train_unique_clean = np.unique(train_clean, axis=0)

# Load validation data
attack_val   = 'pgd'
val_strength = 16
val_step     = val_strength / 10.
if attack_val == 'fgsm':
    loadfile = base_folder + '/%s_val_strength%d.mat' % (attack_val, val_strength)
elif attack_val == 'pgd':
    loadfile = base_folder + '/%s_val_strength%d_step%.2f.mat' % (attack_val, val_strength, val_step)
contents   = hdf5storage.loadmat(loadfile)
val_adv = contents['x_adv_val']
val_labels = to_categorical(np.ones((val_adv.shape[0],)))
# Determine rough number of noisy repetitions for each (to compensate for imbalanced data)
reps_val   = int(np.floor(val_adv.shape[0] / x_det_val.shape[0]) - 1)

# Hyperparameters
k_nearest_range = [10]
batch_size      = 200   
# With the values used in the original source code, but modified for our scale
noise_scale_range = np.asarray([0.0504]) * np.sqrt(boxmax - boxmin)

# Number of validation batches
num_val_adv_batches   = val_adv.shape[0] // batch_size + 1
num_val_clean_batches = x_det_val.shape[0] // batch_size + 1

# Global FPR/TPR/AUC
global_fpr = []
global_tpr = []
global_auc = []

for j, k_nearest in enumerate(k_nearest_range):
    # Get characteristics of validation data
    # Sample a batch of clean data points for each validation sample
    val_adv_characteristics = np.empty((0, 41))
    for batch_idx in tqdm(range(num_val_adv_batches)):
        # get deep representations
        funcs = [K.function([model.layers[0].input, K.learning_phase()], [out])
             for out in get_layer_wise_activations(model, 'cifar')]
        random_idx = np.random.choice(np.arange(train_unique_clean.shape[0]), batch_size, replace=False)
        # Clean training data
        clean_data = train_unique_clean[random_idx]
        # Adversarial validation data
        adv_data   = val_adv[batch_idx*batch_size:np.minimum((batch_idx+1)*batch_size, val_adv.shape[0])]
        
        # Feed
        n_feed = adv_data.shape[0]
        # Local characteristics
        local_val_characteristics = np.zeros((n_feed, 41))
        # Clean activations
        for i, func in enumerate(funcs):
            X_act = func([clean_data[:n_feed], 0])[0]
            X_act = np.asarray(X_act, dtype=np.float32).reshape((n_feed, -1))
            # Adversarial activations
            X_adv_act = func([adv_data, 0])[0]
            X_adv_act = np.asarray(X_adv_act, dtype=np.float32).reshape((n_feed, -1))
            
            # Get characteristics
            local_val_characteristics[:,i] = mle_batch(X_act, X_adv_act, k=k_nearest)
        
        # Append
        val_adv_characteristics = np.append(val_adv_characteristics, local_val_characteristics, axis=0)
    
    # Save characteristics to file
    hdf5storage.savemat('results/val_adv_characteristics_k%d.mat' % (k_nearest),
                        {'val_adv_characteristics': val_adv_characteristics})
    
    # Repeat for clean validation data
    val_clean_characteristics = np.empty((0, 41))
    for batch_idx in tqdm(range(num_val_clean_batches)):
        # get deep representations
        funcs = [K.function([model.layers[0].input, K.learning_phase()], [out])
             for out in get_layer_wise_activations(model, 'cifar')]
        random_idx = np.random.choice(np.arange(train_unique_clean.shape[0]), batch_size, replace=False)
        # Clean training data
        clean_data = train_unique_clean[random_idx]
        # Clean validation data
        clean_val_data = x_det_val[batch_idx*batch_size:np.minimum((batch_idx+1)*batch_size, x_det_val.shape[0])]
        
        # Feed
        n_feed = clean_val_data.shape[0]
        # Local characteristics
        local_val_characteristics = np.zeros((n_feed, 41))
        
        # Clean activations
        for i, func in enumerate(funcs):
            X_act = func([clean_data[:n_feed], 0])[0]
            X_act = np.asarray(X_act, dtype=np.float32).reshape((n_feed, -1))
            # Adversarial activations
            X_clean_val_act = func([clean_val_data, 0])[0]
            X_clean_val_act = np.asarray(X_clean_val_act, dtype=np.float32).reshape((n_feed, -1))
        
            # Get characteristics
            local_val_characteristics[:,i] = mle_batch(X_act, X_clean_val_act, k=k_nearest)
        
        # Append
        val_clean_characteristics = np.append(val_clean_characteristics, local_val_characteristics, axis=0)
    
    # Save characteristics to file
    hdf5storage.savemat('results/val_clean_characteristics_k%d.mat' % (k_nearest),
                        {'val_clean_characteristics': val_clean_characteristics})
    
    # Train different logistic regressions for different noisy data samples
    for i, noise_scale in enumerate(noise_scale_range):
        # Load training data
        contents = hdf5storage.loadmat('results/train_characteristics_k%d_sigma%.6f.mat' % (k_nearest, noise_scale))
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
        
        # Normalize with scaler
        val_adv_characteristics = scaler.transform(val_adv_characteristics)
        val_clean_characteristics = scaler.transform(val_clean_characteristics)
        # Merge and create labels
        merged_val_data = np.concatenate((val_adv_characteristics, val_clean_characteristics), axis=0)
        merged_val_labels = np.concatenate((np.ones((val_adv_characteristics.shape[0],)),
                                            np.zeros((val_clean_characteristics.shape[0],))), axis=0)
        
        # Predict clean and adversarial validation data
        y_val_pred = lr.predict_proba(merged_val_data)
        fpr, tpr, thresholds = roc_curve(merged_val_labels, y_val_pred[:, 1])
        area = auc(fpr, tpr)
        
        global_fpr.append(fpr)
        global_tpr.append(tpr)
        global_auc.append(area)

print('Testing starts here!')
np.error()

# Pick the best values and test with them
winner_k = 10
winner_noise_scale = 0.0504 * np.sqrt(boxmax - boxmin)
test_folder = '../cifar10_blackbox_samples'

# Clean test data
x_det_test = x_test
y_det_test = y_test

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

# Test everything
test_strength_range = [8, 16, 64, 255]

# Number of test batches
num_test_clean_batches = x_det_test.shape[0] // batch_size

# Get characteristics of clean test data
test_clean_characteristics = np.empty((0, 41))
for batch_idx in tqdm(range(num_test_clean_batches)):
    # get deep representations
    funcs = [K.function([model.layers[0].input, K.learning_phase()], [out])
         for out in get_layer_wise_activations(model, 'cifar')]
    random_idx = np.random.choice(np.arange(train_unique_clean.shape[0]), batch_size, replace=False)
    # Clean training data
    clean_data = train_unique_clean[random_idx]
    # Clean test data
    clean_test_data = x_det_test[batch_idx*batch_size:np.minimum((batch_idx+1)*batch_size, x_det_test.shape[0])]
    
    # Feed
    n_feed = clean_test_data.shape[0]
    # Local characteristics
    local_test_characteristics = np.zeros((n_feed, 41))
    
    # Clean activations
    for i, func in enumerate(funcs):
        X_act = func([clean_data[:n_feed], 0])[0]
        X_act = np.asarray(X_act, dtype=np.float32).reshape((n_feed, -1))
        # Adversarial activations
        X_clean_test_act = func([clean_test_data, 0])[0]
        X_clean_test_act = np.asarray(X_clean_test_act, dtype=np.float32).reshape((n_feed, -1))
    
        # Get characteristics
        local_test_characteristics[:,i] = mle_batch(X_act, X_clean_test_act, k=winner_k)
    
    # Append
    test_clean_characteristics = np.append(test_clean_characteristics, local_test_characteristics, axis=0)

# Save characteristics to file
hdf5storage.savemat('results/test_clean_characteristics_k%d.mat' % (winner_k),
                    {'test_clean_characteristics': test_clean_characteristics})

# Transform with scaler
test_clean_characteristics = scaler.transform(test_clean_characteristics)

test_auc = []
test_tpr = []
test_fpr = []

# For each test PGD strength
for test_strength in test_strength_range:
    # Load attacked test data
    attack_test   = 'pgd'
    if test_strength < (255. - 1e-6):
        test_step = test_strength / 10.
    else:
        test_step = 5.
    if attack_test == 'fgsm':
        loadfile = test_folder + '/%s_test_strength%d.mat' % (attack_test, test_strength)
    elif attack_test == 'pgd':
        loadfile = test_folder + '/%s_test_strength%d_step%.2f.mat' % (attack_test, test_strength, test_step)
    contents      = hdf5storage.loadmat(loadfile)
    test_adv      = contents['x_adv_val']
    test_clean    = contents['x_clean_val']
    test_labels   = to_categorical(np.ones((test_adv.shape[0],)))
    
    # Number of test batches
    num_test_adv_batches = test_adv.shape[0] // batch_size + 1
    
    # Get characteristics of adversarial test data
    test_adv_characteristics = np.empty((0, 41))
    for batch_idx in tqdm(range(num_test_adv_batches)):
        # get deep representations
        funcs = [K.function([model.layers[0].input, K.learning_phase()], [out])
             for out in get_layer_wise_activations(model, 'cifar')]
        random_idx = np.random.choice(np.arange(train_unique_clean.shape[0]), batch_size, replace=False)
        # Clean training data
        clean_data = train_unique_clean[random_idx]
        # Adversarial test data
        adv_test_data = test_adv[batch_idx*batch_size:np.minimum((batch_idx+1)*batch_size, test_adv.shape[0])]
        
        # Feed
        n_feed = adv_test_data.shape[0]
        # Local characteristics
        local_test_characteristics = np.zeros((n_feed, 41))
        
        # Clean activations
        for i, func in enumerate(funcs):
            X_act = func([clean_data[:n_feed], 0])[0]
            X_act = np.asarray(X_act, dtype=np.float32).reshape((n_feed, -1))
            # Adversarial activations
            X_adv_test_act = func([adv_test_data, 0])[0]
            X_adv_test_act = np.asarray(X_adv_test_act, dtype=np.float32).reshape((n_feed, -1))
        
            # Get characteristics
            local_test_characteristics[:,i] = mle_batch(X_act, X_adv_test_act, k=winner_k)
        
        # Append
        test_adv_characteristics = np.append(test_adv_characteristics, local_test_characteristics, axis=0)

    # Save characteristics to file
    hdf5storage.savemat('results/test_adv_characteristics_k%d_PGD_strength%d.mat' % (winner_k, test_strength),
                        {'test_adv_characteristics': test_adv_characteristics})
    
    # Transform with scaler
    test_adv_characteristics = scaler.transform(test_adv_characteristics)
    
    # Evaluate together with clean data and get AUC
    merged_data = np.concatenate((test_clean_characteristics, test_adv_characteristics), axis=0)
    merged_labels = np.concatenate((np.zeros((test_clean_characteristics.shape[0],)),
                                    np.ones((test_adv_characteristics.shape[0],))), axis=0)
    # Predict
    y_test_pred = lr.predict_proba(merged_data)
    fpr, tpr, thresholds = roc_curve(merged_labels, y_test_pred[:, 1])
    area = auc(fpr, tpr)
    
    # Save
    test_auc.append(area)
    test_tpr.append(tpr)
    test_fpr.append(fpr)
    