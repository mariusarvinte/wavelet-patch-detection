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

# Where to save
save_dir = 'cifar10_blackbox_samples'

# Attack target
target_data = 'train'
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
# Attack type
attack_type     = 'pgd'
attack_order    = np.inf
attack_strength = 16.
if attack_strength < (255. - 1e-6):
    # Regular
    attack_step = attack_strength / 10.
else:
    # Shrink step size
    attack_step = 5.
attack_masked   = True

# One-shot generation of validation images
num_images  = x_input.shape[0]
batch_size  = 256 # Attack in batches
# Sanity
if batch_size > num_images:
    batch_size = num_images
num_batches = 128 # Sampling without replacement
# Masking parameters
min_size    = [4, 4]
max_size    = [8, 8]

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

# Generate attacked images
x_adv_val   = np.empty((0, 32, 32, 3))
x_clean_val = np.empty((0, 32, 32, 3))
x_mask      = np.empty((0, 32, 32, 3))
for batch_idx in range(num_batches):
    # Extract images
    random_idx   = np.random.choice(np.arange(num_images), size=batch_size)
    local_inputs = x_input[random_idx]
    local_labels = y_input[random_idx]
    # Generate random patch    
    corner_pixel = [np.random.randint(low=0., high=local_inputs.shape[1]-max_size[0], size=(batch_size,)),
                    np.random.randint(low=0., high=local_inputs.shape[2]-max_size[1], size=(batch_size,))]
    patch_size   = [np.random.randint(low=min_size[0], high=max_size[0]+1, size=(batch_size,)),
                    np.random.randint(low=min_size[1], high=max_size[1]+1, size=(batch_size,))]
    # Create mask for each image individually
    mask = np.zeros((batch_size, 32, 32, 3))
    for image_idx in range(batch_size):
        mask[image_idx, corner_pixel[0][image_idx]:corner_pixel[0][image_idx]+patch_size[0][image_idx],
             corner_pixel[1][image_idx]:corner_pixel[1][image_idx]+patch_size[1][image_idx], :] = 1

    # Run attack
    val_adv = attack_images(local_inputs, local_labels, adv_x,
                            x, y, batch_size, sess,
                            attack, mask_tensor, mask)
    
    # Check success and downselect
    y_hat = model.predict(val_adv)
    success_idx = np.where(np.argmax(y_hat, axis=-1) != np.argmax(local_labels, axis=-1))
    # Add to collections
    x_mask    = np.append(x_mask, mask[success_idx], axis=0)
    x_adv_val = np.append(x_adv_val, val_adv[success_idx], axis=0)
    x_clean_val = np.append(x_clean_val, local_inputs[success_idx], axis=0)

# Save to .mat file
savefile = save_dir + '/%s_%s_strength%d_step%.2f.mat' % (attack_type, target_data, attack_strength, attack_step)
hdf5storage.savemat(savefile,
                    {'x_adv_val': x_adv_val,
                     'x_clean_val': x_clean_val,
                     'x_mask': x_mask})