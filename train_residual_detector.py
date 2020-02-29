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
from sklearn.metrics import roc_curve, auc

from cleverhans.utils_keras import KerasModelWrapper

from aux_models import vgg19_model
from aux_evaluation import get_residual
from aux_evaluation import train_classifier, evaluate_classifier

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
boxmin  = (0 - train_mean) / (train_std + 1e-7)
boxmax  = (255 - train_mean) / (train_std + 1e-7)

# Weights
weight_path  = 'models/weights_clean_best.h5'
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
print("Defined TensorFlow model graph.")
# Load weights
model.load_weights(weight_path)
wrap = KerasModelWrapper(model)

# Meta-parameters
train_sigma_range = np.asarray([0.02, 0.05, 0.1, 0.2])
num_epochs        = 400

# Base folder for fetching data
base_folder = 'cifar10_blackbox_samples'

# Training and validation data
batch_size  = 256
num_batches = 16
# Find correctly classified data
y_hat = model.predict(x_val)
valid_idx = np.where(np.argmax(y_hat, axis=-1) == np.argmax(y_val, axis=-1))
x_valid = x_val[valid_idx]
y_valid = y_val[valid_idx]
# Targets
x_det_train = x_valid[:batch_size*num_batches]
y_det_train = y_valid[:batch_size*num_batches]
x_det_val   = x_valid[batch_size*num_batches:]
y_det_val   = y_valid[batch_size*num_batches:]
# And scratches
x_det_train_scratch = np.copy(x_det_train)
x_det_val_scratch   = np.copy(x_det_val)

# Load training data - concatenated
attack_train = 'fgsm'
train_strength_range = [4, 8, 16, 64]
train_adv = np.empty((0, input_w, input_h, input_c))
for train_strength in train_strength_range:
    train_step = train_strength / 10.
    if attack_train == 'pgd':
        loadfile = base_folder + '/%s_train_strength%d_step%.2f.mat' % (attack_train, train_strength, train_step)
    else:
        loadfile = base_folder + '/%s_train_strength%d.mat' % (attack_train, train_strength)
    contents   = hdf5storage.loadmat(loadfile)
    train_adv  = np.append(train_adv, contents['x_adv_val'], axis=0)
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
reps_train = int(np.floor(train_adv.shape[0] / x_det_train.shape[0]) - 1)
reps_val   = int(np.floor(val_adv.shape[0] / x_det_val.shape[0]) - 1)

# Metrics
C_val_collect   = []
tpr_val_collect = []
fpr_val_collect = []
# Run a hyperparameter-search
for train_sigma in train_sigma_range:
    # Reset to scratches
    x_det_train = np.copy(x_det_train_scratch)
    x_det_val   = np.copy(x_det_val_scratch)
          
    # Extract residuals and labels from training images
    for rep_idx in range(reps_train):
        train_noisy  = x_det_train_scratch + np.random.randint(low=-3, high=3, size=x_det_train_scratch.shape) / (train_std + 1e-7)
        x_det_train  = np.concatenate((x_det_train, train_noisy), axis=0)
    train_inputs = np.concatenate((x_det_train, train_adv), axis=0)
    train_labels = np.concatenate((np.zeros((x_det_train.shape[0],)),
                                   np.ones((train_adv.shape[0],))), axis=0)
    train_labels = to_categorical(train_labels)
    train_residuals, train_filtered = get_residual(train_inputs, train_mean, train_std,
                                                   sigma=train_sigma)
    
    # Extract residuals and labels from validation images
    for rep_idx in range(reps_val):
        val_noisy  = x_det_val_scratch + np.random.randint(low=-3, high=3, size=x_det_val_scratch.shape) / (train_std + 1e-7)
        x_det_val  = np.concatenate((x_det_val, val_noisy), axis=0)
    val_inputs = np.concatenate((x_det_val, val_adv), axis=0)
    val_labels = np.concatenate((np.zeros((x_det_val.shape[0],)),
                                 np.ones((val_adv.shape[0],))), axis=0)
    val_labels = to_categorical(val_labels)
    val_residuals, val_filtered = get_residual(val_inputs, train_mean, train_std,
                                                   sigma=train_sigma)
    
    # Pack parameters
    params = dict()
    params['input_w']    = input_w
    params['input_h']    = input_h
    params['input_c']    = input_c
    params['train_data'] = train_residuals
    params['val_data']   = val_residuals
    params['train_labels'] = train_labels
    params['val_labels']   = val_labels
    params['batch_size']   = 32
    params['restore']      = True
    params['num_epochs']   = num_epochs
    params['sigma']        = train_sigma
    
    # Train a classifier on all datapoints
    joint_classifier, _ = train_classifier(params)
    
    # Evaluate
    C_val, tpr_val, fpr_val = evaluate_classifier(joint_classifier, params['val_data'],
                                                  val_labels)
    
    # Store in collection
    C_val_collect.append(C_val)
    tpr_val_collect.append(tpr_val)
    fpr_val_collect.append(fpr_val)

## Testing stage

# Pick a parameter or an ensemble of them
test_sigma = 0.05

# Instantiate a classifier
params = dict()
params['input_w']      = input_w
params['input_h']      = input_h
params['input_c']      = input_c
params['train_data']   = None
params['val_data']     = None
params['train_labels'] = None
params['val_labels']   = None
params['batch_size']   = 128
params['restore']      = False
params['num_epochs']   = 0.
params['sigma']        = test_sigma
# Meta
joint_classifier, _ = train_classifier(params)

# Reload weights
joint_classifier.load_weights('models/weights_best_EXPS_sigma%.3f.h5' % test_sigma)
test_folder = 'cifar10_blackbox_samples'

# Clean test data
x_det_test  = x_test
y_det_test  = y_test
# Load attacked test data
attack_test   = 'pgd'
test_strength = 8 # Change this to other values
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

# Merge
test_noisy     = test_clean + np.random.randint(low=-3, high=3, size=test_clean.shape) / (train_std + 1e-7)
test_negatives = np.concatenate((test_clean, test_noisy), axis=0)
test_inputs    = np.concatenate((test_negatives, test_adv), axis=0)
test_labels    = np.concatenate((np.zeros((test_negatives.shape[0],)),
                             np.ones((test_adv.shape[0],))), axis=0)
test_labels = to_categorical(test_labels)
test_residuals, test_filtered = get_residual(test_inputs, train_mean, train_std,
                              sigma=test_sigma)
test_data = test_residuals

# Evaluate
C_test, tpr_test, fpr_test = evaluate_classifier(joint_classifier, test_data,
                                                 test_labels)
# Generate a ROC curve
y_hat = joint_classifier.predict(test_data)
fpr, tpr, thresholds = roc_curve(test_labels[:,1], y_hat[:,1])
area = auc(fpr, tpr)
plt.figure(); plt.plot(fpr, tpr); plt.grid(); plt.show()