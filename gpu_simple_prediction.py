import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image
import glob
import random
import os
from sklearn.utils import shuffle
import argparse
import math

SHOW_DEBUG_INFO = True
LIMIT_FLAG = False

PROJECT_HOME_DIRECTORY = "/home/" #set home directory/project directory
TEST_IMAGE_DIRECTORY = PROJECT_HOME_DIRECTORY + "Documents/test/101940/"
SAVED_MODEL_PATH = TEST_IMAGE_DIRECTORY+"all_images_conv_Classification_VGG19.h5"

IMAGES_DIRECTORY = PROJECT_HOME_DIRECTORY + "images/"
TRANSLATE_INITIATION_IMAGES = IMAGES_DIRECTORY + "translate_initiation/"
RIGHT_TERMINATION_IMAGES = IMAGES_DIRECTORY + "translateRight/translateRight_termination/"
DEFAULT_INITIATION_IMG_PATH = TRANSLATE_INITIATION_IMAGES + "mobilityBottle3380.png"
DEFAULT_TERMINATION_IMG_PATH = RIGHT_TERMINATION_IMAGES + "mobilityBottle3380-translate-right-0.25.png"


dim=(128,128)
n_channels=3
batch_size=32
snapshot_size = 5
class_categories_dict ={'raise': 0, 'lower': 1, 'translateLeft': 2, 'translateRight': 3, 'push': 4, 'pull': 5, 'open': 6, 'close': 7, 'removePart': 8, 'insertPart': 9, 'removeWhole': 10, 'flip': 11, 'roll': 12, 'turn': 13, 'none3': 14, 'none2':15}

def debug(caption, string):
    if SHOW_DEBUG_INFO == True:
        print(caption, string)
        


def make_prediction(img_path_list, SAVED_MODEL_PATH): 
    """Testing preprocessing and prediction
    
    Inputs: 
    - img_path_list: pass 5 image paths for the 5 multistep from MAB [initiation_path, snap2_path, snap3_path, snap4_path, termination_path]
    - test_labels: list, list of labels that corresponds to the testing images
    - generator_name: str, name for the generator 
    - SAVED_MODEL_PATH: str, path for the model to predict with
    
    Returns: 
    - predictions: list, from the classifier given testing images
    """
    batch_size = 1
    x_test = np.empty((snapshot_size, batch_size, *dim, n_channels))
    ind_counter = 0
    for img_path in img_path_list:
        img = Image.open(img_path)
        rgb = img.convert('RGB')
        new_size = (128, 128)
        rgb = rgb.resize(new_size)
        img_arr = np.asarray(rgb)
        x_test[ind_counter, 0,] = img_arr
        ind_counter += 1
                
    X = []
        # X is array of multi input, for examples, 
    for i in range(snapshot_size):
        X.append(x_test[i])
    
    
    prediction = None        
    saved_model = keras.models.load_model(SAVED_MODEL_PATH)
    
    prediction = saved_model(X, training=False)

    print("prediction:", prediction)
    return prediction  

def cross_entropy(y_true, y_pred):
    """reimplemented cross entropy loss to prevent internal rounding"""
    eps = 1e-45
    y_pred[y_pred < eps] = eps
    cross = np.sum(-y_true * np.log(y_pred), axis=1)
    return cross[0]


def retrieve_score_SCALAR(prediction, verb): 
    """
    Returns the cross_entropy loss given the prediction and desired verb.
    Inputs:
    -prediction: prediction received from neural network
    -verb: desired verb
    
    Outputs: 
    - prediction
    - target vector (all zeros except the actual verb)
    - score: cross-entropy loss between the target array and a single prediction
    """
    sing_prediction = prediction[0]
    print("single_prediction_size:", sing_prediction.shape)
    verb_index = class_categories_dict[verb]
    target = np.zeros(len(class_categories_dict)) #target vector
    target[verb_index] = 1 #assuming prediction vector has the same label ordering as dictionary
    print("target size:", target.shape)

    sw = np.full(len(class_categories_dict), 0.03)
    score = cross_entropy(np.array([target]), np.array([sing_prediction]))
   

    return (sing_prediction, target, score)

def main_without_parameters_SCALAR(img_path_list, saved_model_path, verb):
    """Make prediction, calculates score"""
    prediction = make_prediction(img_path_list, saved_model_path)
    sing_pred, target, score = retrieve_score_SCALAR(prediction, verb)
    return sing_pred, target, score