#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 10:04:35 2020

@author: marius
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras

from keras.utils import to_categorical
from keras.datasets import cifar10

from matplotlib import pyplot as plt
plt.ioff()

from sklearn.model_selection import train_test_split

from aux_models import vgg19_model, detector_model

from aux_evaluation import get_residual

from tqdm import tqdm
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

# Sigma and kappa used for testing
test_sigma = 0.05
test_kappa = 3

# Weights
weight_path = 'models/weights_clean_best.h5'
detector_path = 'models/weights_best_residual_sigma0.050.h5'
# Boxes
boxmin = (0. - train_mean) / (train_std + 1e-7)
boxmax = (255. - train_mean) / (train_std + 1e-7) + 1e-6
# Obtain Image Parameters
img_rows, img_cols, nchannels = x_train.shape[1:4]
nb_classes = y_train.shape[1]

# Classifier
model = vgg19_model(logits=True, img_rows=img_rows, img_cols=img_cols,
                    channels=nchannels,
                    nb_classes=nb_classes)
# Load weights
model.load_weights(weight_path)
# This starts from the residual
detector = detector_model(img_rows=input_h, img_cols=input_w,
                          channels=input_c,
                          nb_classes=2)
print('Defined TensorFlow detector graph.')
# Load weights
detector.load_weights(detector_path)

# Classify and pick clean data
y_hat_clean = model.predict(x_test)
valid_idx   = np.where(np.argmax(y_hat_clean, axis=-1) == np.argmax(y_test, axis=-1))[0]
x_valid     = x_test[valid_idx]
y_valid     = y_test[valid_idx]

# Threat model being evaluated
test_threat_range = ['blackbox', 'graybox', 'whitebox']
distances = dict()
adversaries = dict()

for test_threat in test_threat_range:
    # Folder and filenames
    if test_threat == 'blackbox':
        test_folder = 'results'
        test_name = 'cifarVgg19_CWblackbox_restricted'
        target_name = 'y_target'
    elif test_threat == 'graybox':
        test_folder = 'results'
        test_name = 'cifarVgg19_CWgraybox_restricted'
        target_name = 'y_target_joint'
    elif test_threat == 'whitebox':
        test_folder = 'results'
        test_name = 'cifarVgg19_CWwhitebox_unrestricted'
        target_name = 'y_target_joint'
        
    # Number of patches and other parameters
    if test_threat == 'blackbox' or test_threat == 'graybox':
        test_stage = 10
        test_confidence = 3.
        test_iterations = 10000
    elif test_threat == 'whitebox':
        test_stage = 10
        test_confidence = 3.
        test_iterations = 10000
    
    test_patches = 40
    test_batch   = 256
    # Load all files, record distances and success rate
    success_rate  = np.zeros((test_patches,))
    bbox_success_rate = np.zeros((test_patches,))
    
    # Outputs
    x_adv               = np.empty((0, input_w, input_h, input_c))
    x_black_adv         = np.empty((0, input_w, input_h, input_c))
    x_clean_match       = np.empty((0, input_w, input_h, input_c))
    x_clean_black_match = np.empty((0, input_w, input_h, input_c))
    x_adv_probs         = np.empty((0, num_classes))
    x_mask              = []
    x_mask_idx          = []
    whitebox_logits     = np.empty((0, num_classes))
    
    # Distance matrix across all patches
    global_x_dist = np.zeros((test_batch, test_patches))
    global_images = np.zeros((test_patches, test_batch, input_h, input_w, input_c))
    
    for patch_idx in tqdm(range(test_patches)):
        loadfile = test_folder + '/' + test_name + '_patch%d_iterations%d_confidence%.1f_step%d.mat' % (
                patch_idx, test_iterations, test_confidence, test_stage)
        contents = hdf5storage.loadmat(loadfile)
        # Extract
        local_x_best   = contents['local_x_best']
        local_x_dist   = contents['local_x_dist']
        local_x_clean  = contents['x_clean']
        local_y_clean  = contents['y_clean']
        local_y_target = contents[target_name]
        local_x_pass   = contents['local_x_pass']
        
        # Convert images to hard pixels
        local_x_best = (np.round((local_x_best * (train_std + 1e-7) + train_mean)) - train_mean) / (train_std + 1e-7)
        
        # Find where attack is successful in floating point
        local_success_idx = np.where(local_x_dist < np.inf)[0]
        
        # Add closest distances (normalized) to distance matrix
        global_x_dist[:, patch_idx] = np.sqrt(local_x_dist) * (train_std + 1e-7) / 255.
        # Add images themselves to image matrix
        global_images[patch_idx, :, :, :, :] = local_x_best
    
    # Store
    distances[test_threat]   = np.reshape(global_x_dist.T, (-1))
    adversaries[test_threat] = np.reshape(global_images, (-1, input_h, input_w, input_c))

# Find successful blackbox samples
blackbox_idx     = np.where(distances['blackbox'] < np.inf)[0]
blackbox_success = adversaries['blackbox'][blackbox_idx]
blackbox_dist    = distances['blackbox'][blackbox_idx]
blackbox_ref     = local_x_clean[np.mod(blackbox_idx, test_batch)]
blackbox_ref_y   = local_y_clean[np.mod(blackbox_idx, test_batch)]
# Predict logits
blackbox_logits        = model.predict(blackbox_success)
blackbox_sorted_logits = np.sort(blackbox_logits, axis=-1)
# Compute residuals
blackbox_residuals, _ = get_residual(blackbox_success, train_mean, train_std, 
                                     sigma=test_sigma)
# And detection decision
blackbox_det = detector.predict(blackbox_residuals)
# Find missed detections and where confidence is satisfied
blackbox_missed_det_idx = np.where((np.argmax(blackbox_det, axis=-1) == 0))[0]
blackbox_dist_missed    = blackbox_dist[blackbox_missed_det_idx]

# Find successful graybox examples
graybox_idx     = np.where(distances['graybox'] < np.inf)[0]
graybox_success = adversaries['graybox'][graybox_idx]
graybox_dist    = distances['graybox'][graybox_idx]
graybox_ref     = local_x_clean[np.mod(graybox_idx, test_batch)]
graybox_ref_y   = local_y_clean[np.mod(graybox_idx, test_batch)]
# Predict logits
graybox_logits        = model.predict(graybox_success)
graybox_sorted_logits = np.sort(graybox_logits, axis=-1)
# Select successful classifier examples
graybox_transfer_success_idx   = np.where(np.argmax(graybox_logits, axis=-1) != np.argmax(graybox_ref_y, axis=-1))[0]
graybox_transfer_success       = graybox_success[graybox_transfer_success_idx]
graybox_transfer_sorted_logits = graybox_sorted_logits[graybox_transfer_success_idx]
graybox_dist_success           = graybox_dist[graybox_transfer_success_idx]
# Compute residuals
graybox_residuals, _ = get_residual(graybox_transfer_success, train_mean, train_std, 
                                    sigma=test_sigma)
# And detection decision
graybox_det = detector.predict(graybox_residuals)
# Find missed detections and where confidence is satisfied
graybox_missed_det_idx = np.where((np.argmax(graybox_det, axis=-1) == 0) & 
                                  ((graybox_transfer_sorted_logits[:, -1] -
                                    graybox_transfer_sorted_logits[:, -2]) >= test_kappa))[0]
graybox_dist_missed = graybox_dist_success[graybox_missed_det_idx]

# Find successful whitebox examples
whitebox_idx     = np.where(distances['whitebox'] < np.inf)[0]
whitebox_success = adversaries['whitebox'][whitebox_idx]
whitebox_ref_y   = local_y_clean[np.mod(whitebox_idx, test_batch)]
whitebox_ref     = local_x_clean[np.mod(whitebox_idx, test_batch)]
whitebox_preds   = model.predict(whitebox_success)