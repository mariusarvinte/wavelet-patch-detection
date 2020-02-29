#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 17:22:44 2020

@author: marius
"""

from skimage.restoration import denoise_wavelet
from sklearn.metrics import confusion_matrix

from aux_residual_classifier import ResidueClassifier
from aux_residual_classifier import ResidueLogitClassifier
from aux_residual_classifier import UnknownClassifier
from aux_residual_classifier import BinaryResidueClassifier
from aux_residual_classifier import SmallResidueClassifier

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras import backend as K
import tensorflow as tf

import numpy as np
from tqdm import tqdm

# Custom single logit FPR metric
def binary_fpr_metric(y_true, y_pred):
    # Find locations
    true_negative_idx = K.cast(K.less(y_true, 0.5), 'float32')
    pred_positive_idx = K.cast(K.greater(y_pred, 0.5), 'float32')
    pred_negative_idx = K.cast(K.less(y_pred, 0.5), 'float32')
    
    # Metric
    fpr = K.sum(tf.multiply(pred_positive_idx, true_negative_idx)) / \
        (K.sum(tf.multiply(pred_positive_idx, true_negative_idx)) + \
         K.sum(tf.multiply(pred_negative_idx, true_negative_idx)) + K.epsilon())
    
    return fpr

# Custom TPR metric
def binary_tpr_metric(y_true, y_pred):
    # Find locations
    true_positive_idx = K.cast(K.greater(y_true, 0.5), 'float32')
    pred_positive_idx = K.cast(K.greater(y_pred, 0.5), 'float32')
    pred_negative_idx = K.cast(K.less(y_pred, 0.5), 'float32')
    
    # Metric
    tpr = K.sum(tf.multiply(pred_positive_idx, true_positive_idx)) / \
        (K.sum(tf.multiply(pred_positive_idx, true_positive_idx)) + \
         K.sum(tf.multiply(pred_negative_idx, true_positive_idx)) + K.epsilon())
    
    return tpr

# Custom FPR metric
def fpr_metric(y_true, y_pred):
    # Find locations
    true_negative_idx = K.cast(K.equal(K.argmax(y_true, axis=-1), 0), 'float32')
    pred_positive_idx = K.cast(K.equal(K.argmax(y_pred, axis=-1), 1), 'float32')
    pred_negative_idx = K.cast(K.equal(K.argmax(y_pred, axis=-1), 0), 'float32')
    
    # Metric
    fpr = K.sum(tf.multiply(pred_positive_idx, true_negative_idx)) / \
        (K.sum(tf.multiply(pred_positive_idx, true_negative_idx)) + \
         K.sum(tf.multiply(pred_negative_idx, true_negative_idx)) + K.epsilon())
    
    return fpr

# Custom TPR metric
def tpr_metric(y_true, y_pred):
    # Find locations
    true_positive_idx = K.cast(K.equal(K.argmax(y_true, axis=-1), 1), 'float32')
    pred_positive_idx = K.cast(K.equal(K.argmax(y_pred, axis=-1), 1), 'float32')
    pred_negative_idx = K.cast(K.equal(K.argmax(y_pred, axis=-1), 0), 'float32')
    
    # Metric
    tpr = K.sum(tf.multiply(pred_positive_idx, true_positive_idx)) / \
        (K.sum(tf.multiply(pred_positive_idx, true_positive_idx)) + \
         K.sum(tf.multiply(pred_negative_idx, true_positive_idx)) + K.epsilon())
    
    return tpr

# Generate randomly patched AWGN images
def gaussian_patch_images(inputs, max_size, min_size,
                          train_mean, train_std, eps=1e-7):
    # Outputs
    outputs = np.copy(inputs)
    num_images = inputs.shape[0]
    # Patch each image
    corner_pixel = [np.random.randint(low=0., high=inputs.shape[1]-max_size[0], size=(num_images,)),
                    np.random.randint(low=0., high=inputs.shape[2]-max_size[1], size=(num_images,))]
    patch_size   = [np.random.randint(low=min_size[0], high=max_size[0]+1, size=(num_images,)),
                    np.random.randint(low=min_size[1], high=max_size[1]+1, size=(num_images,))]

    # In a loop
    for image_idx in range(num_images):
        outputs[image_idx,
                corner_pixel[0][image_idx]:corner_pixel[0][image_idx]+patch_size[0][image_idx],
                corner_pixel[1][image_idx]:corner_pixel[1][image_idx]+patch_size[1][image_idx],
                :] = (np.random.randint(low=0, high=256, size=(patch_size[0][image_idx], 
                patch_size[1][image_idx], 3)) - train_mean) / (train_std + 1e-7)

    return outputs
    
# Attack a set of points with a preconstructed graph and tensor
def attack_images(inputs, labels,
                  tensor_out, tensor_in, tensor_labels,
                  batch_size, sess, masked, mask_tensor,
                  mask_numpy):
    # Compute number of batches
    num_batches = int(np.ceil(inputs.shape[0] / batch_size))
    
    # Run attack in batches
    adv = np.empty((0,)+(inputs.shape[1], inputs.shape[2], inputs.shape[3]))
    for batch_idx in tqdm(range(num_batches)):
        # Construct dictionary
        if masked:
            feed_dict = {tensor_in: inputs[batch_idx*batch_size:(batch_idx+1)*batch_size],
                         tensor_labels: labels[batch_idx*batch_size:(batch_idx+1)*batch_size],
                         mask_tensor: mask_numpy}
        else:
            feed_dict = {tensor_in: inputs[batch_idx*batch_size:(batch_idx+1)*batch_size],
                         tensor_labels: labels[batch_idx*batch_size:(batch_idx+1)*batch_size]}
        # Extract adversarial samples
        adv = np.append(adv, tensor_out.eval(session=sess, feed_dict=feed_dict), axis=0)
    
    # Return
    return adv
    
# Compute residuals for a set of inputs
def get_residual(inputs, train_mean, train_std, 
                 sigma, scale=255., eps=1e-7, method='wavelet'):
    # Copy
    filtered = (np.copy(inputs) * (train_std + eps) + train_mean) / scale
    # Denoise with wavelet (or other methods)
    for image_idx in tqdm(range(filtered.shape[0])):
        filtered[image_idx] = \
            (denoise_wavelet(filtered[image_idx], sigma=sigma, 
                             method='BayesShrink', multichannel=True,
                             convert2ycbcr=True) * 255. - train_mean) / (train_std + 1e-7)
        # If any nan value is present, then image is BW and needs to be run without YCBCR conversion
        if np.any(np.isnan(filtered[image_idx])):
            # Restore and recompute
            filtered[image_idx] = (np.copy(inputs[image_idx]) * (train_std + eps) + train_mean) / scale
            filtered[image_idx] = \
                (denoise_wavelet(filtered[image_idx], sigma=sigma, 
                             method='BayesShrink', multichannel=True,
                             convert2ycbcr=False) * 255. - train_mean) / (train_std + 1e-7)
    # Estimate residuals
    residuals = inputs - filtered
    
    return residuals, filtered

# Train and return a residual classifier with given training, validation
# and test data
def train_classifier(params):
    # Unpack everything
    input_w = params['input_w']
    input_h = params['input_h']
    input_c = params['input_c']
    train_data   = params['train_data']
    val_data     = params['val_data']
    train_labels = params['train_labels']
    val_labels   = params['val_labels']
    batch_size   = params['batch_size']
    num_epochs   = params['num_epochs']
    sigma        = params['sigma']
    restore      = params['restore']
    
    # Construct and train the detection NN
    classifier, logit_exposer = ResidueClassifier(input_w, input_h, input_c, weight_reg=5e-3)

    # Loss
    loss = 'categorical_crossentropy'
    metrics = ['acc', fpr_metric, tpr_metric]
    # Optimizer
    optimizer = Adam(lr=0.001, amsgrad=True)
    # Compile
    classifier.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    # Weight file
    weight_file = 'weights_best_sigma%.3f.h5' % sigma
    # Callbacks
    best_model = ModelCheckpoint(weight_file,
                                 verbose=0, save_best_only=True, save_weights_only=True, period=5,
                                 monitor='val_acc')
    early_stop = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=100)
    # Train or not
    if num_epochs > 0.:
        _ = classifier.fit(train_data, train_labels,
                           batch_size=batch_size,
                           epochs=num_epochs,
                           validation_data=(val_data, val_labels),
                           callbacks=[best_model, early_stop],
                           shuffle=True, verbose=2)
    # Load best weights
    if restore:
        classifier.load_weights(weight_file)
    
    return classifier, logit_exposer

# Evaluate a classifier on specific, labeled data and return metrics
def evaluate_classifier(classifier, inputs, labels):
    # Predict
    labels_hat = classifier.predict(inputs)
    # Evaluate
    C   = confusion_matrix(np.argmax(labels, axis=-1), np.argmax(labels_hat, axis=-1))
    tpr = C[1, 1] / (C[1, 1] + C[1, 0])
    fpr = C[0, 1] / (C[0, 1] + C[0, 0])
    
    return C, tpr, fpr

# End-to-end classification, from images to joint-labels
def evaluate_joint_classifier(classifier, detector, kappa,
                              sigma, train_mean, train_std, 
                              inputs):
    # Initially all samples are not detected
    labels_hat = np.zeros((inputs.shape[0],))
    
    # Get classification logits
    class_logits = classifier.predict(inputs)
    
    # Compute image residuals
    residuals, _ = get_residual(inputs, train_mean, train_std, 
                 sigma, scale=255., eps=1e-7, method='wavelet')
    
    # Get detection logits
    det_logits = detector.predict(residuals)
    det_logit  = det_logits[:,1] - det_logits[:,0]
    
    # Mark points where detection is triggered    
    det_idx = np.where(det_logit > 0)[0]
    labels_hat[det_idx] = 1
    
    # Construct additional logit
    new_logit = (1 + det_logit) * np.max(class_logits, axis=-1)
    # Merged logits
    merged_logits = np.concatenate((class_logits, new_logit[:,None]), axis=-1)
    
    # Sort and evaluate difference between logits
    sorted_logits = np.sort(merged_logits, axis=-1)
    # Confidence
    confidence = sorted_logits[:,-1] - sorted_logits[:,2]
    # Mark points where detection fails but confidence is not sufficient
    det_confidence_idx = np.where((det_logit <= 0) & (confidence < kappa))[0]
    labels_hat[det_confidence_idx] = 1
       
    # Construct a sigmoid probability
    det_prob = 1 - 1 / (1 + np.exp(confidence-kappa))
    
    return labels_hat, det_prob