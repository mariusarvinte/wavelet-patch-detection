#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:44:23 2020

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
from aux_models import vgg19_model

from cleverhans.utils_keras import KerasModelWrapper
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

# Weights
weight_path = 'models/weights_clean_best.h5'

# Boxes
boxmin = (0. - train_mean) / (train_std + 1e-7)
boxmax = (255. - train_mean) / (train_std + 1e-7) + 1e-6
# Obtain Image Parameters
img_rows, img_cols, nchannels = x_train.shape[1:4]
nb_classes = y_train.shape[1]

# Define TF model graph for the predictor
model = vgg19_model(img_rows=img_rows, img_cols=img_cols,
                    channels=nchannels,
                    nb_classes=nb_classes)
print('Defined TensorFlow model graph.')
# Load weights and create Keras wrapper
model.load_weights(weight_path)
wrap = KerasModelWrapper(model)
    
# Classify and pick clean data
y_hat_clean = model.predict(x_test)
valid_idx   = np.where(np.argmax(y_hat_clean, axis=-1) == np.argmax(y_test, axis=-1))[0]
x_valid = x_test[valid_idx]
y_valid = y_test[valid_idx]

# Overall confidence
confidence = 3.
# Number of iterations
num_iterations = 10000
# Number of patch locations in each image
num_patches = 40
# Mask bounds
min_size = [6, 6]
max_size = min_size
# Lambda bound and bisection steps
lambda_bounds = [1e-2, 1e10]
lambda_steps  = 11
alpha = 1.
# Special naming
if alpha == 0:
    restricted_name = 'unrestricted'
else:
    restricted_name = 'restricted'

# Image batch size
batch_size = 256

# Input tensor for the image
x = tf.placeholder(tf.float32, (batch_size, input_w, input_h, input_c))
# Input (scalar) tensors for mask dimensions
cornerx = tf.placeholder(tf.int32)
cornery = tf.placeholder(tf.int32)
# Universal
sizex = min_size[0]
sizey = min_size[1]

# Create a modifier tensor
modifier = tf.Variable(np.zeros((batch_size, sizex, sizey, input_c), dtype=np.float32))
# Pad it with zeroes
modifier_padded = tf.pad(modifier, [[0,0], [cornerx, 32-cornerx-sizex], [cornery, 32-cornery-sizey], [0,0]])

# Box auxiliaries that ensure implicit bounding
boxmul    = (boxmax - boxmin) / 2.
boxplus   = (boxmax + boxmin) / 2.
new_image = tf.tanh(modifier_padded + x) * boxmul + boxplus

# Classification for the modified image
logits_mod = wrap.get_logits(new_image)
probs_mod  = wrap.get_probs(new_image)
preds_mod  = tf.argmax(logits_mod, axis=-1)

# Reference tensor for the labels (one-hot)
y = tf.placeholder(tf.float32, (batch_size, num_classes))

# Distance losses for both methods
l2_orig = tf.placeholder(tf.float32, (batch_size, input_w, input_h, input_c))
modifier_loss = tf.reduce_sum(tf.square(new_image - (tf.tanh(l2_orig) * boxmul + boxplus)), [-1, -2, -3])

# Construct a targeted logit loss
logit_true  = tf.reduce_sum((y * logits_mod), axis=-1)
logit_other = tf.reduce_max((1 - y) * logits_mod - (y * 10000.), axis=-1)
max_loss    = tf.maximum(0., -logit_true + logit_other + confidence)
# Lambda tensor
lam = tf.placeholder(tf.float32, (batch_size,))

# Loss function for joint-modifier approach
loss  = lam * max_loss + alpha * modifier_loss
grad, = tf.gradients(loss, modifier)

# Adam optimizer
start_vars = set(x.name for x in tf.global_variables())
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
trainer   = optimizer.minimize(loss, var_list=[modifier])
end_vars = tf.global_variables()
new_vars = [x for x in end_vars if x.name not in start_vars]
init      = tf.global_variables_initializer()
sess.run(init)
# Restore model weights after this
model.load_weights(weight_path)
# Create new initializer
mod_init = tf.variables_initializer(var_list=[modifier]+new_vars)

# Number of images matches batch size
num_images = batch_size

# Construct a batch of images and their rotated targets
x_clean       = x_valid[:num_images]
x_clean_tanh  = np.arctanh((x_clean - boxplus) / boxmul * 0.999999) # For stability
y_clean       = y_valid[:num_images]
class_clean   = np.argmax(y_clean, axis=-1)

# Roll each image to another random class
y_target = np.copy(y_clean)
for image_idx in range(num_images):
    y_target[image_idx] = np.roll(y_clean[image_idx],
                  np.random.randint(low=1, high=10), axis=-1)
# Get natural target class
target_class_idx = np.argmax(y_target, axis=-1)

# For each patch location
for patch_idx in range(num_patches):
    # Generate a mask (same for all images) at all coordinates
    cornerx_np = np.random.randint(low=0, high=input_w-max_size[0]+1)
    cornery_np = np.random.randint(low=0, high=input_h-max_size[1]+1)
    
    # Buffer of best images
    local_x_best = np.copy(x_clean)
    local_x_dist = np.inf * np.ones((num_images,))
    local_x_pass = np.zeros((num_images,))

    # Variable upper/lower bounds
    lower_bounds = lambda_bounds[0] * np.ones((batch_size,))
    upper_bounds = lambda_bounds[1] * np.ones((batch_size,))

    # Start from lower bounds
    batch_lambda = np.copy(lower_bounds)
    
    # For each lambda value
    for step_idx in range(lambda_steps):
        # Reset Adam's state
        sess.run(mod_init)

        # Create a filename
        filename = 'results/cifarVgg19_CWblackbox_%s_patch%d_iterations%d_confidence%.1f_step%d.mat' % (
                restricted_name, patch_idx, num_iterations, confidence, step_idx)
        
        # Copy previous distance values
        past_x_dist = np.copy(local_x_dist)
        
        for i in range(num_iterations):                    
            # Dictionary for modifier attack
            feed_dict_mod = {x: x_clean_tanh,
                             l2_orig: x_clean_tanh,
                             y: y_target,
                             lam: batch_lambda,
                             cornerx: cornerx_np,
                             cornery: cornery_np}
        
            # Run attack for modifier
            _, g_mod, mod_loss, p_mod, np_logits_mod, \
            np_loss, np_max_loss, np_newimg = sess.run([trainer, grad, modifier_loss,
                                      preds_mod, logits_mod,
                                      loss, max_loss, new_image], feed_dict_mod)
           
            # Record best found images
            # Images that are correctly classified as their target with the required confidence
            local_success_idx = np.where(np_max_loss == 0)[0]
            # Extract the distances and replace if better
            candidate_dist = mod_loss[local_success_idx]
            current_dist   = local_x_dist[local_success_idx]
            global_success_idx = local_success_idx[np.where(candidate_dist < current_dist)[0]]
            # Replace
            local_x_best[global_success_idx] = np_newimg[global_success_idx]
            local_x_dist[global_success_idx] = mod_loss[global_success_idx]
            local_x_pass[global_success_idx] = 1.
            
            # Report
            if np.mod(i, num_iterations//10) == 0 and i > 0:
                # Passes
                print('A number of %d images fooled the classifier, with an average L2-distance of %.2f' % (
                        np.sum(local_x_pass), np.mean(local_x_dist[local_x_dist < np.inf])))
                
        # Adjust lambda for each point separately
        for idx in range(batch_size):
            # Improvement
            if local_x_dist[idx] < past_x_dist[idx]:
                upper_bounds[idx] = np.minimum(upper_bounds[idx], batch_lambda[idx])
                # We can be more aggressive
                if upper_bounds[idx] < 1e10:
                    batch_lambda[idx] = (lower_bounds[idx] + upper_bounds[idx]) / 2
            # No improvement
            else:
                # If we're still in the initial solution, decrease lambda by 10x
                # But no more than the absolute lower bound
                lower_bounds[idx] = np.maximum(lower_bounds[idx], batch_lambda[idx])
                if upper_bounds[idx] < 1e10:
                    batch_lambda[idx] = (lower_bounds[idx] + upper_bounds[idx]) / 2
                else:
                    batch_lambda[idx] = batch_lambda[idx] * 10

            
        # Save to .mat file
        hdf5storage.savemat(filename, {'local_x_best': local_x_best,
                                       'local_x_dist': local_x_dist,
                                       'local_x_pass': local_x_pass,
                                       'x_clean': x_clean,
                                       'y_clean': y_clean,
                                       'y_target': y_target,
                                       'cornerx': cornerx_np,
                                       'cornery': cornery_np,
                                       'batch_lambda': batch_lambda})