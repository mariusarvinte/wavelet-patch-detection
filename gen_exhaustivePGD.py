#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 11:48:07 2020

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

from aux_attacks import PatchProjectedGradientDescent
from aux_models import vgg19_model
from cleverhans.utils_keras import KerasModelWrapper

from aux_evaluation import attack_images

import hdf5storage

# Sanity
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(0)
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

# Clip data for sanity
x_train = np.clip(x_train, boxmin, boxmax)
x_val   = np.clip(x_val, boxmin, boxmax)
x_test  = np.clip(x_test, boxmin, boxmax)

# Weights
weight_path  = 'models/weights_clean_best.h5'
# Boxes
boxmin = (0. - train_mean) / (train_std + 1e-7)
boxmax = (255. - train_mean) / (train_std + 1e-7)
# Obtain Image Parameters
img_rows, img_cols, nchannels = x_train.shape[1:4]
nb_classes = y_train.shape[1]

# Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                      nchannels))
y = tf.placeholder(tf.float32, shape=(None, nb_classes))
y_target = tf.placeholder(tf.float32, shape=(None, nb_classes))
# Mask placeholder
mask_tensor = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                                nchannels))

# Define TF model graph
model = vgg19_model(img_rows=img_rows, img_cols=img_cols,
                  channels=nchannels,
                  nb_classes=nb_classes)
print("Defined TensorFlow model graph.")
# Load weights
model.load_weights(weight_path)
wrap = KerasModelWrapper(model)

# Training and validation data
batch_size  = 256
num_batches = 16
# Find correctly classified data
y_hat = model.predict(x_val)
valid_idx = np.where(np.argmax(y_hat, axis=-1) == np.argmax(y_val, axis=-1))
x_valid = x_val[valid_idx]
y_valid = y_val[valid_idx]
# Valid points
x_det_train = x_valid[:batch_size*num_batches]
y_det_train = y_valid[:batch_size*num_batches]
x_det_val   = x_valid[batch_size*num_batches:]
y_det_val   = y_valid[batch_size*num_batches:]

# Attack target
target_data = 'test'
# Cases
if target_data == 'train':
    # Targets
    x_input = x_det_train
    y_input = y_det_train
elif target_data == 'val':
    # Targets
    x_input = x_det_val
    y_input = y_det_val
elif target_data == 'test':
    # Targets
    x_input = x_test
    y_input = y_test

# Where to save
save_dir = 'cifar10_blackbox_samples'

# Masking parameters - fixed size for this, covering approximately 2%
min_size = [5, 5]
max_size = min_size
# One-shot generation of validation images
num_images = x_input.shape[0]
batch_size = 200 # Sample some images without replacement
# Sanity
if batch_size > num_images:
    batch_size = num_images
num_batches = (input_w - min_size[0] + 1) ** 2 # All possible 
random_idx  = np.random.choice(x_input.shape[0], size=(batch_size,))
x_input = x_test[random_idx] # Overwrite
y_input = y_test[random_idx]
# Pre-generate mask coordinates
corner_range = np.arange(input_w - min_size[0] + 1)
cornerx, cornery = np.meshgrid(corner_range, corner_range)
cornerx = cornerx.flatten()
cornery = cornery.flatten()

# Meta-attack type
attack_type   = 'pgd'
attack_order  = np.inf
attack_strength_range = [8, 16, 64, 255]
attack_masked = True

# Success rate
success_rate = []
# Loop
for attack_strength in attack_strength_range:
    # Step size is lower for stronger attack
    if attack_strength < (255. - 1e-6):
        attack_step = attack_strength / 10.
    else:
        attack_step = 5.
    
    # Create attack object
    attack = PatchProjectedGradientDescent(wrap, sess=sess)
    pgd_params = {'eps': attack_strength / 255. * (boxmax - boxmin),
                  'eps_iter': attack_step / 255. * (boxmax - boxmin),
                  'clip_min': boxmin,
                  'clip_max': boxmax,
                  'nb_iter': 100,
                  'rand_init': True,
                  'mask': mask_tensor,
                  'ord': attack_order}
    adv_x = attack.generate(x, **pgd_params)
    adv_x = tf.stop_gradient(adv_x)
    
    # Measure attack success
    success_percentage = 0

    # Generate attacked images
    x_adv_val   = np.empty((0, input_w, input_h, input_c))
    for batch_idx in range(num_batches):
        # No more randomness in inputs or mask
        local_inputs = np.copy(x_input)
        local_labels = np.copy(y_input)
         # Generate a universal mask
        mask = np.zeros((batch_size, input_w, input_h, input_c))
        mask[:, cornerx[batch_idx]:cornerx[batch_idx]+min_size[0],
             cornery[batch_idx]:cornery[batch_idx]+min_size[1], :] = 1
        
        # Run attack
        val_adv = attack_images(local_inputs, local_labels, adv_x,
                                x, y, batch_size, sess,
                                attack, mask_tensor, mask)
        
        # Check success and downselect
        y_hat = model.predict(val_adv)
        success_idx = np.where(np.argmax(y_hat, axis=-1) != np.argmax(local_labels, axis=-1))
        # Record success percentage
        success_percentage += len(success_idx) / batch_size
        # Add to collections
        x_adv_val = np.append(x_adv_val, val_adv, axis=0)
    
    # Average success rate
    success_rate.append(success_percentage/num_batches)
    
    # Save to .mat file
    savefile = save_dir + '/exhaustive_mask%d_%s_%s_strength%d_step%.2f.mat' % (min_size[0], attack_type,
                                                                    target_data, attack_strength, attack_step)
    hdf5storage.savemat(savefile,
                        {'x_adv_val': x_adv_val,
                         'x_clean_val': x_input,
                         'y_clean_val': y_input,
                         'random_idx': random_idx,
                         'success_percentage': success_percentage},
                         truncate_existing=True)