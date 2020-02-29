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
from sklearn.metrics import roc_curve, auc

from aux_models import vgg19_model, detector_model
from cleverhans.utils_keras import KerasModelWrapper

from aux_evaluation import get_residual
from aux_evaluation import evaluate_classifier

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
# Import
from aux_cross_large_networks import CrossVGG19
encoder, logit_exposer, full_activation = CrossVGG19(input_h, input_w, input_c,
                                                      output_dim, num_fc_layers,
                                                      hidden_fc_dim, common_layer,
                                                      output_layer, weight_reg,
                                                      local_seed=0, verbose=False)
# Load weights
encoder.load_weights(weight_path)
# This starts from the residual
detector = detector_model(img_rows=input_h, img_cols=input_w,
                          channels=input_c,
                          nb_classes=2)
print('Defined TensorFlow detector graph.')
# Load weights
detector.load_weights(detector_path)
# Boxes
boxmin = (0. - train_mean) / (train_std + 1e-7)
boxmax = (255. - train_mean) / (train_std + 1e-7) + 1e-6
# Obtain Image Parameters
img_rows, img_cols, nchannels = x_train.shape[1:4]
nb_classes = y_train.shape[1]

# Classify and pick clean data
y_hat_clean = encoder.predict(x_test)
valid_idx   = np.where(np.argmax(y_hat_clean, axis=-1) == np.argmax(y_test, axis=-1))[0]
x_valid     = x_test[valid_idx]
y_valid     = y_test[valid_idx]

# Threat model being evaluated
test_threat_range = ['white']
distances = dict()
adversaries = dict()

for test_threat in test_threat_range:
    # Folder and filenames
    if test_threat == 'black':
        test_folder = 'results'
        test_name = 'cifarVgg19_CWblackbox_restricted'
        target_name = 'y_target'
    elif test_threat == 'gray':
        test_folder = 'results'
        test_name = 'cifarVgg19_CWgraybox_restricted'
        target_name = 'y_target_joint'
    elif test_threat == 'white':
        test_folder = 'results'
        test_name = 'cifarVgg19_CWwhitebox_unrestricted'
        target_name = 'y_target_joint'
        
    # Number of patches and other parameters
    if test_threat == 'black' or test_threat == 'gray':
        test_stage = 10
        test_confidence = 3.
        test_iterations = 10000
    elif test_threat == 'white':
        test_stage = 0
        test_confidence = 3.
        test_iterations = 5000
    
    test_patches    = 40
    test_batch      = 256
    
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
black_idx     = np.where(distances['black'] < np.inf)[0]
black_success = adversaries['black'][black_idx]
black_dist    = distances['black'][black_idx]
black_ref     = local_x_clean[np.mod(black_idx, test_batch)]
black_ref_y   = local_y_clean[np.mod(black_idx, test_batch)]
# Predict logits
black_logits  = logit_exposer.predict(black_success)
black_sorted_logits = np.sort(black_logits, axis=-1)
# Compute residuals
black_residuals, _ = get_residual(black_success, train_mean, train_std, 
                                 sigma=test_sigma)
# And detection decision
black_det = detector.predict(black_residuals)
# Find missed detections and where confidence is satisfied
black_missed_det_idx = np.where((np.argmax(black_det, axis=-1) == 0))[0]
black_dist_missed = black_dist[black_missed_det_idx]

# Find successful graybox examples
gray_idx     = np.where(distances['gray'] < np.inf)[0]
gray_success = adversaries['gray'][gray_idx]
gray_dist    = distances['gray'][gray_idx]
gray_ref     = local_x_clean[np.mod(gray_idx, test_batch)]
gray_ref_y   = local_y_clean[np.mod(gray_idx, test_batch)]
# Predict logits
gray_logits  = logit_exposer.predict(gray_success)
gray_sorted_logits = np.sort(gray_logits, axis=-1)
# Select successful classifier examples
gray_transfer_success_idx = np.where(np.argmax(gray_logits, axis=-1) != np.argmax(gray_ref_y, axis=-1))[0]
gray_transfer_success = gray_success[gray_transfer_success_idx]
gray_transfer_sorted_logits  = gray_sorted_logits[gray_transfer_success_idx]
gray_dist_success = gray_dist[gray_transfer_success_idx];
# Compute residuals
gray_residuals, _ = get_residual(gray_transfer_success, train_mean, train_std, 
                                 sigma=test_sigma)
# And detection decision
gray_det = detector.predict(gray_residuals)
# Find missed detections and where confidence is satisfied
gray_missed_det_idx = np.where((np.argmax(gray_det, axis=-1) == 0) & 
                               ((gray_transfer_sorted_logits[:, -1] - gray_transfer_sorted_logits[:, -2]) >= test_kappa))[0]
gray_dist_missed = gray_dist_success[gray_missed_det_idx]

# Find successful whitebox examples
white_idx     = np.where(distances['white'] < np.inf)[0]
white_success = adversaries['white'][white_idx]
white_ref_y   = local_y_clean[np.mod(white_idx, test_batch)]
white_ref     = local_x_clean[np.mod(white_idx, test_batch)]
white_preds   = encoder.predict(white_success)
# Add noise
white_noisy   = white_success + np.random.randint(low=-20, high=21, size=white_success.shape) * (boxmax-boxmin) / 255
white_noisy_y = encoder.predict(white_noisy)

# Sort distances
#sorted_white_idx  = np.argsort(distances['white'])
# Anomalies
anomaly_idx = np.where((distances['black'] < np.inf) & (distances['white'] < np.inf) &
                       (distances['white'] < distances['black'] - 0.5) & (distances['white'] < 1.5))[0]
#anomaly_idx = sorted_white_idx[:8]
# Get classes of anomalies
clean_y = np.argmax(encoder.predict(local_x_clean[np.mod(anomaly_idx, test_batch)]), axis=-1)
black_y = np.argmax(encoder.predict(adversaries['black'][anomaly_idx]), axis=-1)
white_y = np.argmax(encoder.predict(adversaries['white'][anomaly_idx]), axis=-1)

# Plot anomalies
for idx, img_idx in enumerate(anomaly_idx):
    plt.figure();
    plt.subplot(1, 3, 1); plt.imshow((local_x_clean[np.mod(img_idx, test_batch)] * (train_std + 1e-7) + train_mean) / 255.);
    plt.title('Clean: %s' % (class_name[clean_y[idx]]))
    plt.subplot(1, 3, 2); plt.imshow((adversaries['black'][img_idx] * (train_std + 1e-7) + train_mean) / 255.);
    plt.title('Blackbox: %s' % (class_name[black_y[idx]]))
    plt.subplot(1, 3, 3); plt.imshow((adversaries['white'][img_idx] * (train_std + 1e-7) + train_mean) / 255.);
    plt.title('Whitebox: %s' % (class_name[white_y[idx]]))
    plt.savefig('anomaly%d.png' % img_idx, dpi=300)
    plt.close()
    
# Get all activations
full_clean = full_activation.predict(local_x_clean[np.mod(anomaly_idx, test_batch)])
full_black = full_activation.predict(adversaries['black'][anomaly_idx])
full_white = full_activation.predict(adversaries['white'][anomaly_idx])
# Pick an image and a set of filters
image_idx = 1
filter_idx = 0
filter_clean = full_clean[filter_idx][image_idx]
filter_black = full_black[filter_idx][image_idx]
filter_white = full_white[filter_idx][image_idx]
for local_idx in range(filter_clean.shape[-1]):
    plt.figure()
    plt.subplot(1, 3, 1); plt.imshow(filter_clean[:, :, local_idx]);
    plt.subplot(1, 3, 2); plt.imshow(filter_black[:, :, local_idx]);
    plt.subplot(1, 3, 3); plt.imshow(filter_white[:, :, local_idx]);
    plt.savefig('anomaly%d_filter%d.png' % (image_idx, local_idx), dpi=300)
    plt.close()
    
    
# Apply a number of random masks and extract classifications
num_images = x_valid.shape[0]
num_patches = 20
min_size    = [6, 6]
max_size    = min_size
# Outputs
patched_y = np.zeros((num_images, num_classes, num_patches))
for patch_idx in range(num_patches):
    # Generate a mask (same for all images) at all coordinates
    cornerx = np.random.randint(low=0, high=input_w-max_size[0]+1)
    cornery = np.random.randint(low=0, high=input_h-max_size[1]+1)
    sizex   = np.random.randint(low=min_size[0], high=max_size[0]+1)
    sizey   = np.random.randint(low=min_size[1], high=max_size[1]+1)
    grad_mask = np.zeros((num_images, input_w, input_h, input_c))
    grad_mask[:, cornerx:cornerx+sizex, cornery:cornery+sizey, :] = 1
    
    # Generate patched images + uniform pixel noise
    patched_x = (1 - grad_mask) * x_valid + grad_mask * np.random.randint(
            low=0, high=256, size=x_valid.shape) * (boxmax - boxmin) / 255.
    # Classify and collect
    patched_y[:, :, patch_idx] = encoder.predict(patched_x)
    
# Patch high-confidence whitebox adversaries
num_images  = white_success.shape[0]
num_patches = 50
min_size    = [6, 6]
max_size    = min_size
# Outputs
patched_y_adv = np.zeros((num_images, num_classes, num_patches))
for patch_idx in range(num_patches):
    # Generate a mask (same for all images) at all coordinates
    cornerx = np.random.randint(low=0, high=input_w-max_size[0]+1)
    cornery = np.random.randint(low=0, high=input_h-max_size[1]+1)
    sizex   = np.random.randint(low=min_size[0], high=max_size[0]+1)
    sizey   = np.random.randint(low=min_size[1], high=max_size[1]+1)
    grad_mask = np.zeros((num_images, input_w, input_h, input_c))
    grad_mask[:, cornerx:cornerx+sizex, cornery:cornery+sizey, :] = 1
    
    # Generate patched images + uniform pixel noise
    patched_x_adv = (1 - grad_mask) * white_success + grad_mask * np.random.randint(
            low=0, high=256, size=white_success.shape) * (boxmax - boxmin) / 255.
    # Classify and collect
    patched_y_adv[:, :, patch_idx] = encoder.predict(patched_x_adv)
    

from grad_cam import grad_cam, guided_backprop, build_guided_model
from grad_cam import deprocess_image

# Anomalies
anomaly_idx = np.where((distances['black'] < np.inf))[0]

# Guided model
guided_encoder = build_guided_model(encoder)
# Get anomaly predictions
anomaly_x = adversaries['black'][anomaly_idx]
clean_x   = local_x_clean[np.mod(anomaly_idx, test_batch)]
clean_y   = local_y_clean[np.mod(anomaly_idx, test_batch)]
anomaly_y_hat = encoder.predict(anomaly_x)
anomaly_class = np.argmax(anomaly_y_hat, axis=-1)
# Where to look
layer_name = 'block3_conv4'
# Downselect
img_idx = 180
clean_img = clean_x[img_idx]
adv_img   = anomaly_x[img_idx]
# Another class
clean_class = np.argmax(clean_y[img_idx])
adv_class   = np.argmax(anomaly_y_hat[img_idx])

# GB maps - class agnostic
clean_gb = guided_backprop(guided_encoder, clean_img, layer_name)
adv_gb = guided_backprop(guided_encoder, adv_img, layer_name)

plt.ioff()
plt.figure()
# Plot images
plt.subplot(2, 11, 1); plt.imshow((clean_img * (train_std + 1e-7) + train_mean) / 255.); plt.title('Clean image')
plt.subplot(2, 11, 0+12); plt.imshow((adv_img * (train_std + 1e-7) + train_mean) / 255.); plt.title('Fake image')
# Grad cams and Guided Grad Cams
for class_idx in range(num_classes):
    clean_cam = grad_cam(encoder, clean_img, class_idx, layer_name)
    clean_gcam = clean_gb * clean_cam[..., np.newaxis]
    plt.subplot(2, 11, class_idx+2)
    plt.imshow(clean_cam)
    if class_idx != clean_class:
        plt.axis('off')
    
    adv_cam = grad_cam(encoder, adv_img, class_idx, layer_name)
    adv_gcam = adv_gb * adv_cam[..., np.newaxis]
    plt.subplot(2, 11, class_idx+13)
    plt.imshow(adv_cam)
    if class_idx != adv_class:
        plt.axis('off')

plt.savefig('all_class_activation_map.png', dpi=300)
plt.close()

# Plot
plt.figure();
plt.subplot(241); plt.imshow((clean_img * (train_std + 1e-7) + train_mean) / 255.); plt.title('Clean image')
plt.subplot(242); plt.imshow(np.flip(deprocess_image(clean_true_gcam), -1));
plt.title('True class: %s' % class_name[clean_class])
plt.subplot(243); plt.imshow(np.flip(deprocess_image(clean_fake_gcam), -1));
plt.title('Fake class: %s' % class_name[adv_class])
plt.subplot(244); plt.imshow(np.flip(deprocess_image(clean_third_gcam), -1));
plt.title('Third class: %s' % class_name[third_class])

plt.subplot(245); plt.imshow((adv_img * (train_std + 1e-7) + train_mean) / 255.); plt.title('Fake image')
plt.subplot(246); plt.imshow(np.flip(deprocess_image(adv_true_gcam), -1));
plt.title('True class: %s' % class_name[clean_class])
plt.subplot(247); plt.imshow(np.flip(deprocess_image(adv_fake_gcam), -1));
plt.title('Fake class: %s' % class_name[adv_class])
plt.subplot(248); plt.imshow(np.flip(deprocess_image(adv_third_gcam), -1));
plt.title('Third class: %s' % class_name[third_class])
plt.show()

from keras.applications.mobilenet_v2 import MobileNetV2

#    # Classify all
#    local_adv_class  = model.predict(local_x_best)
#    local_adv_logits = shadow_model.softmax_exposer.predict(local_x_best)
#    
#    # Compute all residuals
#    local_residuals_adv_pixels, _ = get_residual(local_x_best[local_success_idx], train_mean, train_std,
#                                           sigma=test_sigma)
#    adv_detect = classifier.predict(local_residuals_adv_pixels)    
#    adv_whitebox_success_idx = local_success_idx[np.where((np.argmax(local_adv_class[local_success_idx], axis=-1) != np.argmax(local_y_clean[local_success_idx], axis=-1)) &
#                                        (np.argmax(adv_detect, axis=-1) == 0))[0]]
#    
#    # Get logits of whitebox examples
#    local_whitebox_logits = local_adv_logits[adv_whitebox_success_idx]
#    whitebox_logits = np.append(whitebox_logits, local_whitebox_logits, axis=0)
#    
#    # Separate success for classifier only
#    adv_blackbox_success_idx = local_success_idx[np.where((np.argmax(local_adv_class[local_success_idx], axis=-1) ==
#                                                           np.argmax(local_y_target[local_success_idx], axis=-1)))[0]]
#    
#    # Success rate
#    success_rate[patch_idx] = len(adv_whitebox_success_idx) / local_x_clean.shape[0]
#    bbox_success_rate[patch_idx] = len(adv_blackbox_success_idx) / local_x_clean.shape[0]
#    
#    # Terminate early
#    if len(adv_whitebox_success_idx) == 0 and len(adv_blackbox_success_idx) == 0:
#        continue
#    
#    x_mask.append(local_x_mask)
#    x_mask_idx.append(patch_idx * np.ones((adv_whitebox_success_idx.shape[0],)))
#    # Perform a sanity check
#    sanity_clean_labels = model.predict(local_x_clean[adv_whitebox_success_idx])
#    sanity_adv_labels   = model.predict(local_x_best[adv_whitebox_success_idx])
#    label_clean_match   = np.mean(np.argmax(local_y_clean[adv_whitebox_success_idx], axis=-1) == 
#                                  np.argmax(sanity_clean_labels, axis=-1))
#    label_mismatch      = np.mean(np.argmax(sanity_adv_labels, axis=-1) == 
#                                  np.argmax(sanity_clean_labels, axis=-1))
#    # Raise errors if needed
#    if label_clean_match < 1 - 1e-10:
#        print('ERROR: Clean samples do not have 100% classification accuracy!')
#        np.error()
#
#    # Add to collections
#    x_black_adv   = np.append(x_black_adv, local_x_best[np.where(local_x_pass)[0]], axis=0)
#    x_adv         = np.append(x_adv, local_x_best[adv_whitebox_success_idx], axis=0)
#    x_adv_probs   = np.append(x_adv_probs, local_adv_class[adv_whitebox_success_idx], axis=0)
#    x_clean_match = np.append(x_clean_match, local_x_clean[adv_whitebox_success_idx], axis=0)
#    x_clean_black_match = np.append(x_clean_black_match, local_x_clean[np.where(local_x_pass)[0]], axis=0)
#
## Bring images to [0, 1] range and round to pixels
#x_adv_norm = np.round((x_adv * (train_std + 1e-7) + train_mean)) / 255.
#x_black_adv_norm = np.round((x_black_adv * (train_std + 1e-7) + train_mean)) / 255.
#x_clean_match_norm = np.round((x_clean_match * (train_std + 1e-7) + train_mean)) / 255.
#x_clean_black_match_norm = np.round((x_clean_black_match * (train_std + 1e-7) + train_mean)) / 255.
## Compute average distance
#x_dist = np.sqrt(np.sum(np.square(x_adv_norm - x_clean_match_norm), axis=(-3, -2, -1)))
#x_black_dist = np.sqrt(np.sum(np.square(x_black_adv_norm - x_clean_black_match_norm), axis=(-3, -2, -1)))

np.error()

# Compute whitebox confidence
whitebox_logits_sorted = np.sort(whitebox_logits, axis=-1)
whitebox_confidence = whitebox_logits_sorted[:, -1] - whitebox_logits_sorted[:, -2]
# How many examples evade detection
kappa = 3
whitebox_success_idx = np.where(whitebox_confidence > kappa)[0]
whitebox_success = len(whitebox_success_idx) / (test_batch * test_patches)
whitebox_success_dist = x_dist[whitebox_success_idx]

# Recompute new adversary
x_adv_pixels = (x_adv_norm * 255. - train_mean) / (train_std + 1e-7)
# Classify and detect
adv_class  = model.predict(x_adv_pixels)
# Compute residuals
residuals_adv_pixels, _ = get_residual(x_adv_pixels, train_mean, train_std,
                                       sigma=test_sigma)
adv_detect = classifier.predict(residuals_adv_pixels)
# Final succesful samples
adv_whitebox_success_idx = np.where((np.argmax(adv_class, axis=-1) == np.argmax(local_y_target, axis=-1)) &
                                    (np.argmax(adv_detect, axis=-1) == 0))[0]

# Create noisy data
test_noisy  = x_valid + np.random.randint(low=-3, high=3, size=x_valid.shape) / (train_std + 1e-7)
x_det_test  = np.concatenate((x_valid, test_noisy), axis=0)
# Merge
test_inputs = np.concatenate((x_det_test, x_adv), axis=0)
test_labels = np.concatenate((np.zeros((x_det_test.shape[0],)),
                             np.ones((x_adv.shape[0],))), axis=0)
test_labels = to_categorical(test_labels)    
test_residuals = get_residual(test_inputs, train_mean, train_std,
                              sigma=test_sigma)
test_logits = shadow_model.softmax_exposer.predict(test_inputs)
test_data   = test_residuals
# Evaluate
C_test, tpr_test, fpr_test = evaluate_classifier(classifier, test_data,
                                                 test_labels)
# Generate a ROC curve
y_hat = classifier.predict(test_data)
fpr, tpr, thresholds = roc_curve(test_labels[:,1], y_hat[:,1])
area = auc(fpr, tpr)
plt.figure(); plt.plot(fpr, tpr); plt.grid(); plt.show()


# For each patch, compute average L2 distance in it and create a map
dist_heatmap    = np.zeros((input_h, input_w))
success_heatmap = np.zeros((input_h, input_w))
patch_dist = np.zeros((test_patches,))
running_idx = 0
for patch_idx in range(test_patches):
    valid_length = len(x_mask_idx[patch_idx])
    # Distance in that patch
    patch_dist[patch_idx] = np.mean(x_dist[running_idx:running_idx+valid_length])
    # Find corner
    local_mask = x_mask[patch_idx][0] # NOTE: We assume this has copied masks inside
    row_ones = [np.where(local_mask[idx] == 1)[0] for idx in range(input_h)]
    found_ones = [len(row_ones[idx]) > 0 for idx in range(input_h)]
    row_idx = np.where(found_ones)[0][0]
    col_idx = row_ones[row_idx][0]
    # Mark map with average and success rate
    dist_heatmap[row_idx, col_idx] = patch_dist[patch_idx]
    success_heatmap[row_idx, col_idx] = success_rate[patch_idx]
    # Increment
    running_idx = running_idx + valid_length

# Plot two maps
plt.figure();
plt.subplot(121); plt.imshow(dist_heatmap, cmap='hot'); 
plt.colorbar(fraction=0.046, pad=0.04); plt.title('L2-distance')
plt.subplot(122); plt.imshow(success_heatmap, cmap='hot'); 
plt.colorbar(fraction=0.046, pad=0.04); plt.title('Success rate')

# TODO: Show examples of successful adversaries with low/high distances to illustrate difficulties