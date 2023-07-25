import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
import tensorflow_addons as tfa
from keras import Sequential
from keras.optimizers import Adam
from keras.models import load_model
from keras import Sequential
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau
import collections
from PIL import Image
import glob
import random
from sklearn.utils import shuffle
from keras import backend as K 
import re
import argparse
import sys
import time
import traceback
import shutil
from datetime import datetime
import json

from jan2023_batch_data_generator import DataGenerator
## home project directories ##
PROJECT_HOME_DIRECTORY = "/home/rara/Documents/research_2022/" #set to your home directory
## image directories for dataset ##
IMAGES_DIRECTORY = PROJECT_HOME_DIRECTORY + "images/"

CLASSIFIER_EXE_LOG_TXT_PATH = PROJECT_HOME_DIRECTORY + "Classifier_Train_Test_Log.txt"
OPEN_INITIATION_IMAGES = IMAGES_DIRECTORY+"open/open_initiation/"
OPEN_TERMINATION_IMAGES = IMAGES_DIRECTORY+ "open/open_termination/"
RAISE_TERMINATION_IMAGES = IMAGES_DIRECTORY + "raise/raise_termination/"
LOWER_TERMINATION_IMAGES = IMAGES_DIRECTORY + "lower/lower_termination/"
RIGHT_TERMINATION_IMAGES = IMAGES_DIRECTORY + "translateRight/translateRight_termination/"
LEFT_TERMINATION_IMAGES = IMAGES_DIRECTORY + "translateLeft/translateLeft_termination/"
PUSH_TERMINATION_IMAGES = IMAGES_DIRECTORY + "push/push_termination/"
PULL_TERMINATION_IMAGES = IMAGES_DIRECTORY + "pull/pull_termination/"
TRANSLATE_INITIATION_IMAGES = IMAGES_DIRECTORY + "translate_initiation/"
REMOVEPART_TERMINATION_IMAGES = IMAGES_DIRECTORY + "removePart/removePart_termination/"
REMOVEPART_INITIATION_IMAGES = IMAGES_DIRECTORY + "removePart/removePart_initiation/"
REMOVEWHOLE_TERMINATION_IMAGES = IMAGES_DIRECTORY + "removeWhole/removeWhole_termination/"
REMOVEWHOLE_INITIATION_IMAGES = IMAGES_DIRECTORY + "removeWhole/removeWhole_initiation/"
ROTATEROLLCW_TERMINATION_IMAGES = IMAGES_DIRECTORY + "rotateRollCW/rotateRollCW_termination/"
ROTATEROLLCW_INITIATION_IMAGES = IMAGES_DIRECTORY + "rotateRollCW/rotateRollCW_initiation/"
ROTATEROLLCCW_TERMINATION_IMAGES = IMAGES_DIRECTORY + "rotateRollCCW/rotateRollCCW_termination/"
ROTATEROLLCCW_INITIATION_IMAGES = IMAGES_DIRECTORY + "rotateRollCCW/rotateRollCCW_initiation/"
ROTATEPITCHCW_TERMINATION_IMAGES = IMAGES_DIRECTORY + "rotatePitchCW/rotatePitchCW_termination/"
ROTATEPITCHCW_INITIATION_IMAGES = IMAGES_DIRECTORY + "rotatePitchCW/rotatePitchCW_initiation/"
ROTATEPITCHCCW_TERMINATION_IMAGES = IMAGES_DIRECTORY + "rotatePitchCCW/rotatePitchCCW_termination/"
ROTATEPITCHCCW_INITIATION_IMAGES = IMAGES_DIRECTORY + "rotatePitchCCW/rotatePitchCCW_initiation/"
ROTATEYAWCW_TERMINATION_IMAGES = IMAGES_DIRECTORY + "rotateYawCW/rotateYawCW_termination/"
ROTATEYAWCW_INITIATION_IMAGES = IMAGES_DIRECTORY + "rotateYawCW/rotateYawCW_initiation/"
ROTATEYAWCCW_TERMINATION_IMAGES = IMAGES_DIRECTORY + "rotateYawCCW/rotateYawCCW_termination/"
ROTATEYAWCCW_INITIATION_IMAGES = IMAGES_DIRECTORY + "rotateYawCCW/rotateYawCW_initiation/"

########################TODO: NONE#########################
#initiation_directory_list = [OPEN_INITIATION_IMAGES,OPEN_TERMINATION_IMAGES,  
#                            REMOVEPART_INITIATION_IMAGES, REMOVEPART_TERMINATION_IMAGES, REMOVEWHOLE_INITIATION_IMAGES, 
#                            ROTATEROLLCW_INITIATION_IMAGES, ROTATEROLLCCW_INITIATION_IMAGES, 
#                            ROTATEPITCHCW_INITIATION_IMAGES, ROTATEPITCHCCW_INITIATION_IMAGES,
#                            ROTATEYAWCW_INITIATION_IMAGES, ROTATEYAWCCW_INITIATION_IMAGES, TRANSLATE_INITIATION_IMAGES,NONE_INITIATION_IMAGES]
########################TODO: NONE#########################
termination_directory_list = [OPEN_TERMINATION_IMAGES, OPEN_INITIATION_IMAGES, REMOVEPART_TERMINATION_IMAGES, 
                              REMOVEPART_INITIATION_IMAGES, REMOVEWHOLE_TERMINATION_IMAGES, 
                              ROTATEROLLCW_TERMINATION_IMAGES, ROTATEROLLCCW_TERMINATION_IMAGES, 
                              ROTATEPITCHCW_TERMINATION_IMAGES, ROTATEPITCHCCW_TERMINATION_IMAGES, 
                              ROTATEYAWCW_TERMINATION_IMAGES, ROTATEYAWCCW_TERMINATION_IMAGES]


ORIGINAL_ROLLCW = "rotateRollCW"
ORIGINAL_ROLLCCW = "rotateRollCCW"
ORIGINAL_PITCHCW = "rotatePitchCW"
ORIGINAL_PITCHCCW = "rotatePitchCCW"
ORIGINAL_YAWCW = "rotateYawCW"
ORIGINAL_YAWCCW = "rotateYawCCW"


CONVERT_ROLLCW = "roll"
CONVERT_ROLLCCW = "roll"
CONVERT_PITCHCW = "flip"
CONVERT_PITCHCCW = "flip"
CONVERT_YAWCW = "turn"
CONVERT_YAWCCW = "turn"

rotate_convert_labels = [(ORIGINAL_ROLLCW, CONVERT_ROLLCW),
(ORIGINAL_ROLLCCW,CONVERT_ROLLCCW),
(ORIGINAL_PITCHCW,CONVERT_PITCHCW),
(ORIGINAL_PITCHCCW, CONVERT_PITCHCCW),
(ORIGINAL_YAWCW, CONVERT_YAWCW),
(ORIGINAL_YAWCCW, CONVERT_YAWCCW)]

translate_initiation_directory_list = [TRANSLATE_INITIATION_IMAGES]
translate_termination_directory_list = [RAISE_TERMINATION_IMAGES, LOWER_TERMINATION_IMAGES, RIGHT_TERMINATION_IMAGES, LEFT_TERMINATION_IMAGES, PUSH_TERMINATION_IMAGES, PULL_TERMINATION_IMAGES]

rotate_termination_directory_list = [ROTATEROLLCW_TERMINATION_IMAGES, ROTATEROLLCCW_TERMINATION_IMAGES, ROTATEPITCHCW_TERMINATION_IMAGES, ROTATEPITCHCCW_TERMINATION_IMAGES, ROTATEYAWCW_TERMINATION_IMAGES, ROTATEYAWCCW_TERMINATION_IMAGES]
## object categories lists ##

all_object_categories_list = ["Bottle", "Box", "Bucket", "Camera", "Cart", "Chair", "Clock", "CoffeeMachine", "Dishwasher", 
                                    "Dispenser", "Display", "Door", "Eyeglasses", "Fan", "Faucet", "FoldingChair", "Globe", "Kettle",
                                    "Keyboard", "KitchenPot", "Knife", "Lamp", "Laptop", "Lighter", "Microwave", "Mouse", "Oven", "Pen", 
                                    "Phone", "Phone", "Pliers", "Printer", "Refrigerator", "Remote", "Safe", "Scissors", "Stapler", 
                                    "StorageFurniture", "Suitcase", "Switch", "Table", "Toaster", "Toilet", "TrashCan", "USB", "WashingMachine", "Window"]
open_close_categories_list = ["Box", "Dishwasher", "Door", "Laptop", "Microwave", "Oven", "Refrigerator", "Safe", "Stapler", "StorageFurniture", 
                        "Toilet", "TrashCan", "WashingMachine"]
removePart_insertPart_object_categories_list = ["Bottle", "Box", "Bucket", "Camera", "Chair", "Clock", "CoffeeMachine", "Dishwasher", 
                                    "Dispenser", "Display", "Door", "Eyeglasses", "Faucet", "FoldingChair", "Globe", "Kettle",
                                    "KitchenPot", "Knife", "Lamp", "Laptop", "Lighter", "Microwave", "Mouse", "Oven", "Pen", 
                                    "Phone", "Phone", "Pliers", "Printer", "Refrigerator", "Remote", "Safe", "Scissors", "Stapler", 
                                    "StorageFurniture", "Suitcase", "Switch", "Table", "Toaster", "Toilet", "TrashCan", "USB", "WashingMachine", "Window"]
#to be replaced
train_object_categories_list = []

class_categories_dict = {"raise":0, "lower":1, "translateLeft":2, "translateRight":3, "push":4, "pull":5}

## Training Constants ##

ROTATION_AUGMENT = [30, 45, 90]
#5 snapshots
SNAPSHOT_STEPS = ["initiation", "step5", "step10", "step15", "termination"]

TRAIN_RATIO = 0.8
LIMIT_FLAG = False
CUT_OFF_TRAIN_COUNT = 600
CUT_OFF_VAL_COUNT = 200
TEST_IMAGE_LIMIT = 10000
SHOW_DEBUG_INFO = False
BATCH_SIZE = 32
USE_BATCH_SIZE = 32*30 #960

## Global variables which will be edited in prepare_data function ##
class_categories_dict = {"raise":0, "lower":1, "translateLeft":2, "translateRight":3, "push":4, "pull":5}

g_USE_BATCH = False
g_debug_test_variable_info = ""
NUM_EPOCHS = 5
#Verb Flags
OPEN_FLAG = True 
REMOVEPART_FLAG = True
REMOVEWHOLE_FLAG = True
ROTATE_VERB_FLAG = True

########################TODO: NONE#########################



#augmentatio flags
AUGMENT_FLAG = True
FLIP_FLAG = True
ROTATE_FLAG = False

test_translate_categories_list = ["USB", "WashingMachine", "Window"]
test_open_categories_list = ["Box"]
test_removePart_categories_list = ["Door"]
test_removeWhole_categories_list = ["Door"]
test_rotate_categories_list = ["USB", "WashingMachine", "Window"]
test_none_categories_list = ["Door"]
saved_filename = "Rachel_temp_file_name" 
directory_name = "Rachel_test_temp_data_path" 
directory_with_slash = directory_name+"/"

noneVerbDict = None

## paths to save model, training pictures (accuracy and loss), and prediction txts##
SAVED_MODEL_PATH = directory_with_slash+saved_filename+".h5"
SAVED_LOSS_VAL_LOSS_PATH = directory_with_slash+"LossVal_loss_"+saved_filename+".png"
SAVED_ACC_VAL_ACC_PATH = directory_with_slash+"AccVal_acc_"+saved_filename+".png"


PREDICTIONS_TXT_PATH = directory_with_slash+"predictions_"+saved_filename+".txt"
PREDICTIONS_RESULTS_TXT_PATH = directory_with_slash+"predictions_results_"+saved_filename+".txt"
TRANSLATE_PREDICTIONS_TXT_PATH = directory_with_slash+"translate_predictions_"+saved_filename+".txt"
TRANSLATE_PREDICTIONS_RESULTS_TXT_PATH = directory_with_slash+"translate_predictions_results_"+saved_filename+".txt"


OPEN_CLOSE_PREDICTIONS_TXT_PATH = directory_with_slash+"open_close_predictions_"+saved_filename+".txt"
OPEN_CLOSE_PREDICTIONS_RESULTS_TXT_PATH = directory_with_slash+"open_close_predictions_results_"+saved_filename+".txt"
REMOVEPART_INSERTPART_PREDICTIONS_TXT_PATH = directory_with_slash+"removePart_insertPart_predictions_"+saved_filename+".txt"
REMOVEPART_INSERTPART_PREDICTIONS_RESULTS_TXT_PATH = directory_with_slash+"removePart_insertPart_predictions_results_"+saved_filename+".txt"
REMOVEWHOLE_PREDICTIONS_TXT_PATH = directory_with_slash+"removeWhole_predictions_"+saved_filename+".txt"
REMOVEWHOLE_PREDICTIONS_RESULTS_TXT_PATH = directory_with_slash+"removeWhole_predictions_results_"+saved_filename+".txt"
ROTATE_PREDICTIONS_TXT_PATH = directory_with_slash+"rotate_predictions_"+saved_filename+".txt"
ROTATE_PREDICTIONS_RESULTS_TXT_PATH = directory_with_slash+"rotate_predictions_results_"+saved_filename+".txt"

########################none verb#########################
NONE_VERB_FLAG = False
NONE_PREDICTIONS_TXT_PATH = None
NONE_PREDICTIONS_RESULTS_TXT_PATH = None
NONE_TERMINATION_IMAGES = None

TEST_VARIABLE_TXT_PATH = directory_with_slash+"test_variable_"+saved_filename+".txt"

def debug(caption, string):
    """debug function: if SHOW_DEBUG_INFO is true, prints a debug message
    
    Inputs:
    - caption: string, message to be printed
    - string: variable to be printed with the caption
    
    Returns:
    - None
    """
    if SHOW_DEBUG_INFO == True:
        print(caption, string)

def check_object_category(filename):
    """check the object category of the file
    
    Inputs: 
    - filename: str, filepath
    
    Returns:
    - category: str, object category of the object in the file
    """
    basename = os.path.basename(filename)
    filename_split_dashes = basename.split("-")
    upper_case_split = re.findall('[A-Z][a-z]*', filename_split_dashes[0])
    debug("upper_case_split:", upper_case_split)
    category = ''.join(upper_case_split)
    return category

def check_presence_test_object(object_category, term_dir):
    """check if the object category belongs in the test categories for the verbs
    
    Inputs: 
    - object_category: str, object category of file
    - term_dir: str, path of the termination directory that the file belongs in
    
    Returns: 
    - True: if object category is to be tested
    - False: if object category is not to be tested (belongs to the train categories)
    """
    if term_dir in translate_termination_directory_list:
        if not(object_category in test_translate_categories_list):
            return False
        else:
            return True
    elif term_dir== OPEN_TERMINATION_IMAGES: #open/close
        if not(object_category in test_open_categories_list): 
            return False
        else:
            return True
    elif term_dir == REMOVEPART_TERMINATION_IMAGES:
        if not(object_category in test_removePart_categories_list): 
            return False
        else:
            return True
    elif term_dir == REMOVEWHOLE_TERMINATION_IMAGES: 
        if not(object_category in test_removeWhole_categories_list): 
            return False
        else:
            return True
    elif term_dir in rotate_termination_directory_list: 
        if not(object_category in test_rotate_categories_list):
                return False
        else:
            return True

    ########################TODO: NONE#########################
    elif term_dir in NONE_TERMINATION_IMAGES:
        if not(object_category in test_none_categories_list):
            return False
        else:
            return True
    else: 
        raise("Invalid Verb")
    

def find_initiation_path(term_dir, filename):
    """find initiation path 
    
    Inputs: 
    - term_dir: str, filepath of termination directory
    - filename: str, filepath of termination file image
    
    Returns:
    - initiation_image_path: str, filepath of corresponding initiation file image
    """
    initiation_image_path = ""
    if term_dir in translate_termination_directory_list:
        initiation_dir_path = TRANSLATE_INITIATION_IMAGES
        #print("initiation_dir_path:", initiation_dir_path)
        fname = os.path.basename(filename)
        #print("fname:", fname)
        parse_list = fname.split("-")
        initiation_image_path = initiation_dir_path + parse_list[0]+".png"
    else:
        initiation_dir = term_dir.replace('termination', 'initiation')
        base_name = os.path.basename(filename)
        init_filename = base_name.replace('termination.png', 'initiation.png')
        initiation_image_path = initiation_dir + init_filename
    #debug("initiation_image_path:", initiation_image_path)
    return initiation_image_path



def find_label(term_dir):
    """find verb label given directory
    
    Inputs: 
    - term_dir: str, path of termination directory
    
    Returns:
    - pair_label: str, verb label associated with the images
    """
    folder_list = term_dir.split("/")
    current_dir = folder_list[len(folder_list)-2]
    pair_label = current_dir.replace("_termination", "")
    debug("pair_label:", pair_label) 
    return pair_label


def train_validation_files(filenames, labels, categories, TRAIN_RATIO):
    """for MULTI BATCH, split into train and validation filename groups
    
    Inputs: 
    - filenames: list, 
    - labels: list, labels associated with the files
    - categories: list, categories associated with the files
    - TRAIN_RATIO: constant, decimal: percentage of the files to put in training batch, remaining to validation
    
    Returns: 
    - train_filenames: list, filenames for training
    - train_labels: list, labels for the corresponding files for training
    - train_categories: list, categories for the corresponding files for training
    - val_filenames: list, filenames for validation
    - val_labels: list, labels for the corresponding files for validation
    - val_categories: list, categories for the corresponding files for validation
    
    
    """
    number_train = int(len(labels)*TRAIN_RATIO)
    print("X_train:", number_train)
    train_filenames = filenames[:number_train]
    train_labels = labels[:number_train]
    train_categories = categories[:number_train]
    count_label_train = collections.Counter(train_labels)
    print("count_label_train:", count_label_train)
    print("X_val:", len(labels)-number_train)
    #val_set

    val_filenames = filenames[number_train:]
    val_labels = labels[number_train:]
    val_categories = categories[number_train:]
    count_label_val = collections.Counter(val_labels)
    print("count_label_val:", count_label_val)
    return train_filenames, train_labels, train_categories, val_filenames, val_labels, val_categories

def check_prediction(predictions, actual_labels):
    """Checks predictions
    
    Inputs: 
    - predictions: output from prediction from the classifier
    - actual_labels: list, actual labels for the images in test
    
    Returns: 
    - correct_counter: number of images predicted correctly
    - incorrect_counter: number of images prediction incorrectly
    - correct_percentage: percentage of images against total that are correctly predicted
    """
    correct_counter = 0
    incorrect_counter = 0
    data_size = min(predictions.shape[0], len(actual_labels))    
    for x in range(data_size):
        im_prediction = predictions[x]
        predicted_label = np.argmax(im_prediction)
        debug("predicted_label:",predicted_label)
        real_label = actual_labels[x]
        debug("real_label:", real_label)
        real_index = class_categories_dict[real_label]
        debug("real_index:", real_index)
        if predicted_label == real_index:
            correct_counter += 1
        else: 
            incorrect_counter += 1
    correct_percentage = correct_counter/(correct_counter+incorrect_counter)
    print("correct_counter:", correct_counter)
    print("incorrect_counter:", incorrect_counter)
    print("correct_percentage:", correct_percentage)
    return correct_counter, incorrect_counter,correct_percentage

def convert_str_to_bool(s):
    """converts string arguments from the command arguments to Booleans
    
    Inputs: 
    - s: string of Boolean passed from command argument
    
    Returns: 
    Boolean
    """
    if s.lower() == "true":
        return True
    elif s.lower() == "false":
        return False
    else:
        sys.exit("Invalid Boolean:"+s)
        
            

def read_command_line_arg():
    """Read and arg parses arguments from the command line
    
    
    Arguments: 
    - USE_BATCH: String of Boolean, whether to use batches
    - EPOCHS: int, number of training epochs
    - OPEN_CLOSE: String of Boolean, whether to incorporate Open and Close 
    - FLIP: String of Boolean, whether to incorporate flip augmentation
    - ROTATE: String of Boolean, whether to incorporate rotate augmentation
    - Test_Translate_Categories_List: list, list of test categories for translate verbs
    - Test_Open_Close_Categories_List: list, list of test categories for open and close verbs
    """
    
    ########################TODO: NONE#########################
    parser = argparse.ArgumentParser()
    try:
        parser.add_argument('-USE_BATCH', '--useBatch', type=str, help='True or False for using batches in network', default="True")
        parser.add_argument('-EPOCHS', '--epochsNum', type=int, help="Number of EPOCHS, defaault is 40", default = 1)
        parser.add_argument('-OPEN_CLOSE', '--openFlag', help="OPEN_FLAG is True/False, default is True", default = "True")
        parser.add_argument('-REMOVE_PART_INSERT_PART', '--removePartFlag', type=str, help="REMOVEPART_FLAG flag is True/False, default is TRUE", default = "True")
        parser.add_argument('-REMOVE_WHOLE', '--removeWholeFlag', type=str, help="REMOVEWHOLE_FLAG flag is True/False, default is TRUE", default = "True")
        parser.add_argument('-ROTATE_VERB', '--rotateVerbFlag', type=str, help="ROTATE_VERB_FLAG flag is True/False, default is TRUE", default = "True")
        parser.add_argument('-NONE_VERB', '--noneVerbFlag', type=str, help="NONE_VERB_FLAG flag is True/False, default is TRUE", default = "True")
        parser.add_argument('-FLIP', '--flipFlag', type=str, help="FLIP flag is True/False, default is False", default = "True")
        parser.add_argument('-ROTATE', '--rotateFlag', type=str, help="ROTATE flag is True/False, default is False", default = "False")
        parser.add_argument('-Train_Categories_List', '--trainObjectList', type=str, help="train_object_categories_list, for example:USB,WashingMachine,Window",required = False, default = "Box,Dishwasher,Door,Laptop,Microwave,Oven,Refrigerator,Safe,Stapler,StorageFurniture,Toilet,TrashCan,WashingMachine")
        parser.add_argument('-Test_Translate_Categories_List', '--testTranslateList', type=str, help="test_translate_categories_list, for example:USB,WashingMachine,Window",required = False, default = "Toilet")
        parser.add_argument('-Test_Open_Close_Categories_List', '--testOpenList', type=str, help="test_open_categories_list, for example:Box,Toilet",required = False, default = "Toilet")
        parser.add_argument('-Test_RemovePart_InsertPart_Categories_List', '--testRemovePartList', type=str, help="Test_RemovePart_InsertPart_Categories_List, for example:Box,Toilet",required = False, default = "Toilet")
        parser.add_argument('-Test_RemoveWhole_Categories_List', '--testRemoveWholeList', type=str, help="Test_RemoveWhole_Categories_List, for example:Box,Toilet",required = False, default = "Toilet")
        parser.add_argument('-Test_Rotate_Categories_List', '--testRotateList', type=str, help="Test_Rotate_Categories_List, for example:Box,Toilet",required = False, default = "Toilet")
        parser.add_argument('-Test_None_Categories_List', '--testNoneList', type=str, help="Test_None_Categories_List, for example:Box,Toilet",required = False, default = "Toilet")
        parser.add_argument('-SNAPSHOT_STEPS', '--snapshot_steps', type=str, help="steps for multistep, for example: initiation,step1,termination", required=False, default = "initiation,step5,step10,step15,termination")
        #parser.add_argument('-None_Verbs_JSON', '--noneVerbJSON', type=str, help="noneVerbsJson with name and folder",required = False, default ='{"none3":"/home/rara/Documents/research_2022/images/none3/none3_termination/", "none2":"/Users/f1/Documents/CS/images/none2/none2_termination/", "none1": "/Users/f1/Documents/CS/images/none1/none1_termination/"}')
        parser.add_argument('-None_Verbs_JSON', '--noneVerbJSON', type=str, help="noneVerbsJson with name and folder",required = False, default ='{"none3":"/home/rara/Documents/research_2022/images/none3/none3_termination/", "none2":"/home/rara/Documents/research_2022/images/none2/none2_termination/"}')
        #parser.add_argument('-None_Verbs_JSON', '--noneVerbJSON', type=str, help="noneVerbsJson with name and folder",required = False, default ='{"none3":"/gpfs/home/rma20/data/rma20/images/none3/none3_termination/", "none2":"/gpfs/home/rma20/data/rma20/images/none2/none2_termination/"}')
        
        args = parser.parse_args()
        print("--------input data ---------")
        print(f'USE_BATCH: {args.useBatch}')
        print(f'EPOCHS: {args.epochsNum}')
        print(f'OPEN_CLOSE: {args.openFlag}')
        print(f'REMOVE_PART_INSERT_PART: {args.removePartFlag}')
        print(f'REMOVE_WHOLE: {args.removeWholeFlag}')
        print(f'ROTATE_VERB: {args.rotateVerbFlag}')
        print(f'NONE_VERB: {args.noneVerbFlag}')
        
        print(f'FLIP: {args.flipFlag}')
        print(f'ROTATE: {args.rotateFlag}')
        #print(f'scanLibrary: {args.scanLibrary}')
        print(f'Train_Categories_list:{args.trainObjectList}')
        print(f'Test_Translate_Categories_List: {args.testTranslateList}')
        print(f'Test_Open_Close_Categories_List: {args.testOpenList}')
        print(f'Test_RemovePart_InsertPart_Categories_List: {args.testRemovePartList}')
        print(f'Test_RemoveWhole_Categories_List: {args.testRemoveWholeList}')
        print(f'Test_Rotate_Categories_List: {args.testRotateList}')
        print(f'Test_None_Categories_List: {args.testNoneList}')
        print(f'SNAPSHOT_STEPS:{args.snapshot_steps}')
        print(f'noneVerbJSON:{args.noneVerbJSON}')

        global g_USE_BATCH
        global NUM_EPOCHS
        global OPEN_FLAG
        global REMOVEPART_FLAG
        global REMOVEWHOLE_FLAG
        global ROTATE_VERB_FLAG
        global NONE_VERB_FLAG        
        global AUGMENT_FLAG
        global FLIP_FLAG
        global ROTATE_FLAG
        global train_object_categories_list
        global test_translate_categories_list
        global test_open_categories_list
        global test_removePart_categories_list
        global test_removeWhole_categories_list
        global test_rotate_categories_list
        global SNAPSHOT_STEPS
        ########################none verb#########################
        global test_none_categories_list
        global noneVerbDict
        #########################################################

        
        g_USE_BATCH = convert_str_to_bool(args.useBatch)
        NUM_EPOCHS = args.epochsNum
        OPEN_FLAG = convert_str_to_bool(args.openFlag)
        REMOVEPART_FLAG = convert_str_to_bool(args.removePartFlag)
        REMOVEWHOLE_FLAG = convert_str_to_bool(args.removeWholeFlag)
        ROTATE_VERB_FLAG = convert_str_to_bool(args.rotateVerbFlag)
        NONE_VERB_FLAG = convert_str_to_bool(args.noneVerbFlag)
        FLIP_FLAG = convert_str_to_bool(args.flipFlag)
        ROTATE_FLAG = convert_str_to_bool(args.rotateFlag)
        AUGMENT_FLAG = FLIP_FLAG or ROTATE_FLAG
        trainObjectCategoriesListStr = args.trainObjectList
        testTranslateListStr = args.testTranslateList
        testOpenListStr = args.testOpenList
        testRemovePartListStr = args.testRemovePartList
        testRemoveWholeListStr = args.testRemoveWholeList
        testRotateVerbListStr = args.testRotateList
        testNoneVerbListStr = args.testNoneList
        snapshotStepsStr = args.snapshot_steps
        train_object_categories_list = trainObjectCategoriesListStr.replace("\r","").replace("\n","").split(",")
        test_translate_categories_list = testTranslateListStr.replace("\r","").replace("\n","").split(",")
        test_open_categories_list = testOpenListStr.replace("\r","").replace("\n","").split(",")
        test_removePart_categories_list = testRemovePartListStr.replace("\r","").replace("\n","").split(",")
        test_removeWhole_categories_list = testRemoveWholeListStr.replace("\r","").replace("\n","").split(",")
        test_rotate_categories_list = testRotateVerbListStr.replace("\r","").replace("\n","").split(",")
        SNAPSHOT_STEPS = snapshotStepsStr.replace("\r","").replace("\n","").split(",")
        ########################none_verb#########################
        test_none_categories_list = testNoneVerbListStr.replace("\r","").replace("\n","").split(",")
        #none_verb
        if NONE_VERB_FLAG == True:
            noneVerbDict = json.loads(args.noneVerbJSON)
        #########################################################
        
        return

    except Exception: 
        parser.print_help()
        traceback.print_exc()
        sys.exit(0)

def prepare_data():
    """Updates according to the command arguments
    
    Inputs: 
    None
    
    Returns: 
    None
    """
    global saved_filename
    global directory_name
    global directory_with_slash
    global SAVED_MODEL_PATH
    global SAVED_LOSS_VAL_LOSS_PATH
    global SAVED_ACC_VAL_ACC_PATH
    global PREDICTIONS_TXT_PATH
    global PREDICTIONS_RESULTS_TXT_PATH
    global TRANSLATE_PREDICTIONS_TXT_PATH
    global TRANSLATE_PREDICTIONS_RESULTS_TXT_PATH
    global OPEN_CLOSE_PREDICTIONS_TXT_PATH
    global OPEN_CLOSE_PREDICTIONS_RESULTS_TXT_PATH
    global REMOVEPART_INSERTPART_PREDICTIONS_TXT_PATH
    global REMOVEPART_INSERTPART_PREDICTIONS_RESULTS_TXT_PATH
    global REMOVEWHOLE_PREDICTIONS_TXT_PATH
    global REMOVEWHOLE_PREDICTIONS_RESULTS_TXT_PATH
    global ROTATE_PREDICTIONS_TXT_PATH
    global ROTATE_PREDICTIONS_RESULTS_TXT_PATH
    global NONE_PREDICTIONS_TXT_PATH
    global NONE_PREDICTIONS_RESULTS_TXT_PATH


    ########################none_verb#########################
    global TEST_VARIABLE_TXT_PATH 
    global class_categories_dict
    global noneVerbDict
    global NONE_TERMINATION_IMAGES
    ########################################################
    
    print ("--------prepare data ---------")
    print("train_object_categories_list:", train_object_categories_list)
    print("test_translate_categories_list:",test_translate_categories_list)
    print("test_open_categories_list:",test_open_categories_list)
    print("test_removePart_categories", test_removePart_categories_list)
    print("test_removeWhole_categories", test_removeWhole_categories_list)
    print("test_rotate_categories", test_rotate_categories_list)
    print("test_none_categories", test_none_categories_list)
    if OPEN_FLAG == True: 
        current_dict_size = len(class_categories_dict)
        class_categories_dict["open"] = current_dict_size
        class_categories_dict["close"] = current_dict_size + 1
    if REMOVEPART_FLAG == True:
        current_dict_size = len(class_categories_dict)
        class_categories_dict["removePart"] = current_dict_size
        class_categories_dict["insertPart"] = current_dict_size + 1
    if REMOVEWHOLE_FLAG == True: 
        current_dict_size = len(class_categories_dict)
        class_categories_dict["removeWhole"] = current_dict_size
    if ROTATE_VERB_FLAG == True: 
        rotate_label_list = [CONVERT_ROLLCW, CONVERT_ROLLCCW, CONVERT_PITCHCW, CONVERT_PITCHCCW,
                             CONVERT_YAWCW, CONVERT_YAWCCW]
        rotate_set = set(rotate_label_list)
        list_rotate_set = list(rotate_set)
        list_rotate_set.sort()
        current_dict_size = len(class_categories_dict)
        label_key_idx = current_dict_size
        for rotate_element in list_rotate_set:
            class_categories_dict[rotate_element] = label_key_idx
            label_key_idx += 1

    test_categories_name = "translateremoveWholerotatenone" + '-'.join(test_translate_categories_list) + "_openremovePart-" + '-'.join(test_open_categories_list)+ "_removeWhole-" + "-".join(test_removeWhole_categories_list)
    print("test_categories_name:",test_categories_name)
    string_ints = [str(int) for int in ROTATION_AUGMENT]
    rotation_argment_name = "rotation_argment-" + ''.join(string_ints)
    print("rotation_argment_name:",rotation_argment_name)
    
    saved_filename = 'Classification_conv'+"_"+str(NUM_EPOCHS)+"_flip-"+str(FLIP_FLAG)+ "_rotate-"+ str(ROTATE_FLAG)+"_"+str(len(SNAPSHOT_STEPS))+"steps"+"_"+rotation_argment_name+"_"+test_categories_name
    directory_name = PROJECT_HOME_DIRECTORY+saved_filename
    directory_with_slash = directory_name+"/"
    ########################none_verb#########################
    NONE_PREDICTIONS_TXT_PATH = []
    NONE_PREDICTIONS_RESULTS_TXT_PATH = []
    NONE_TERMINATION_IMAGES = []
    if NONE_VERB_FLAG == True:
        current_dict_size = len(class_categories_dict)
        i = 0
        for k in noneVerbDict.keys():
            class_categories_dict[k] = current_dict_size + i
            termination_directory_list.append(noneVerbDict[k])
            NONE_TERMINATION_IMAGES.append(noneVerbDict[k])
            NONE_PREDICTIONS_TXT_PATH.append("{0}{1}_predictions_{2}.txt".format(directory_with_slash, k, saved_filename))
            NONE_PREDICTIONS_RESULTS_TXT_PATH.append("{0}{1}_predictions_results_{2}.txt".format(directory_with_slash, k, saved_filename))
            i += 1
    ########################################################

    SAVED_MODEL_PATH = directory_with_slash+saved_filename+".h5"
    SAVED_LOSS_VAL_LOSS_PATH = directory_with_slash+"LossVal_loss_"+saved_filename+".png"
    SAVED_ACC_VAL_ACC_PATH = directory_with_slash+"AccVal_acc_"+saved_filename+".png"
    PREDICTIONS_TXT_PATH = directory_with_slash+"predictions_"+saved_filename+".txt"
    PREDICTIONS_RESULTS_TXT_PATH = directory_with_slash+"predictions_results_"+saved_filename+".txt"
    TRANSLATE_PREDICTIONS_TXT_PATH = directory_with_slash+"translate_predictions_"+saved_filename+".txt"
    TRANSLATE_PREDICTIONS_RESULTS_TXT_PATH = directory_with_slash+"translate_predictions_results_"+saved_filename+".txt"
    OPEN_CLOSE_PREDICTIONS_TXT_PATH = directory_with_slash+"open_close_predictions_"+saved_filename+".txt"
    OPEN_CLOSE_PREDICTIONS_RESULTS_TXT_PATH = directory_with_slash+"open_close_predictions_results_"+saved_filename+".txt"
    REMOVEPART_INSERTPART_PREDICTIONS_TXT_PATH = directory_with_slash+"removePart_insertPart_predictions_"+saved_filename+".txt"
    REMOVEPART_INSERTPART_PREDICTIONS_RESULTS_TXT_PATH = directory_with_slash+"removePart_insertPart_predictions_results_"+saved_filename+".txt"
    REMOVEWHOLE_PREDICTIONS_TXT_PATH = directory_with_slash+"removeWhole_predictions_"+saved_filename+".txt"
    REMOVEWHOLE_PREDICTIONS_RESULTS_TXT_PATH = directory_with_slash+"removeWhole_predictions_results_"+saved_filename+".txt"
    ROTATE_PREDICTIONS_TXT_PATH = directory_with_slash+"rotate_predictions_"+saved_filename+".txt"
    ROTATE_PREDICTIONS_RESULTS_TXT_PATH = directory_with_slash+"rotate_predictions_results_"+saved_filename+".txt"

    TEST_VARIABLE_TXT_PATH = directory_with_slash+"test_variable_"+saved_filename+".txt"

    return 

def debug_and_print(printStr):
    print(printStr)
    global g_debug_test_variable_info
    g_debug_test_variable_info += printStr + "\n"


def display_and_save_variable():
    print ("--------display variable ---------")

    debug_and_print("USE_BATCH:{0}".format(g_USE_BATCH))
    debug_and_print("NUM_EPOCHS:{0}".format(NUM_EPOCHS))
    debug_and_print("BATCH_SIZE:{0}".format(BATCH_SIZE))
    debug_and_print("OPEN_FLAG:{0}".format(OPEN_FLAG))
    debug_and_print("REMOVEPART_FLAG:{0}".format(REMOVEPART_FLAG))
    debug_and_print("REMOVEWHOLE_FLAG:{0}".format(REMOVEWHOLE_FLAG))
    debug_and_print("ROTATE_VERB_FLAG:{0}".format(ROTATE_VERB_FLAG))
    ########################TODO: NONE#########################
    debug_and_print("NONE_VERB_FLAG:{0}".format(NONE_VERB_FLAG))
    debug_and_print("noneVerbDict:{0}".format(noneVerbDict))

    debug_and_print("FLIP_FLAG:{0}".format(FLIP_FLAG))
    debug_and_print("ROTATE_FLAG:{0}".format(ROTATE_FLAG))
    debug_and_print("AUGMENT_FLAG:{0}".format(AUGMENT_FLAG))
    debug_and_print("SNAPSHOT_STEPS:{0}".format(','.join(SNAPSHOT_STEPS)))
    debug_and_print("number of object categories:{0}".format(len(all_object_categories_list)))
    debug_and_print("number of train object categories:{0}".format(len(train_object_categories_list)))
    debug_and_print("number of translate test object categories:{0}".format(len(test_translate_categories_list)))
    debug_and_print("number of open/close test object categories:{0}".format(len(test_open_categories_list)))
    debug_and_print("train_object_categories_list:{0}".format(','.join(train_object_categories_list)))
    debug_and_print("test_translate_categories_list:{0}".format(','.join(test_translate_categories_list)))
    debug_and_print("test_open_categories_list:{0}".format(','.join(test_open_categories_list)))
    debug_and_print("test_removePart_categories_list:{0}".format(','.join(test_removePart_categories_list)))
    debug_and_print("test_removeWhole_categories_list:{0}".format(','.join(test_removeWhole_categories_list)))
    debug_and_print("test_rotate_categories_list:{0}".format(','.join(test_rotate_categories_list)))
    ########################TODO: NONE#########################
    debug_and_print("test_none_categories_list:{0}".format(','.join(test_none_categories_list)))
    debug_and_print("class_categories_dict:{0}".format(str(class_categories_dict)))
    

    print("directory_name:",directory_name)
    print("SAVED_MODEL_PATH:",SAVED_MODEL_PATH)
    print("SAVED_LOSS_VAL_LOSS_PATH:",SAVED_LOSS_VAL_LOSS_PATH)
    print("SAVED_ACC_VAL_ACC_PATH:",SAVED_ACC_VAL_ACC_PATH)
    print("PREDICTIONS_TXT_PATH:",PREDICTIONS_TXT_PATH)
    print("PREDICTIONS_RESULTS_TXT_PATH:",PREDICTIONS_RESULTS_TXT_PATH)
    print("TRANSLATE_PREDICTIONS_TXT_PATH", TRANSLATE_PREDICTIONS_TXT_PATH)
    print("TRANSLATE_PREDICTIONS_RESULTS_TXT_PATH", TRANSLATE_PREDICTIONS_RESULTS_TXT_PATH)
    print("OPEN_CLOSE_PREDICTIONS_TXT_PATH:",OPEN_CLOSE_PREDICTIONS_TXT_PATH)
    print("OPEN_CLOSE_PREDICTIONS_RESULTS_TXT_PATH:",OPEN_CLOSE_PREDICTIONS_RESULTS_TXT_PATH)
    print("REMOVEPART_INSERTPART_PREDICTIONS_TXT_PATH", REMOVEPART_INSERTPART_PREDICTIONS_TXT_PATH)
    print("REMOVEPART_INSERTPART_PREDICTIONS_RESULTS_TXT_PATH", REMOVEPART_INSERTPART_PREDICTIONS_RESULTS_TXT_PATH)
    print("REMOVEWHOLE_PREDICTIONS_TXT_PATH", REMOVEWHOLE_PREDICTIONS_TXT_PATH)
    print("REMOVEWHOLE_PREDICTIONS_RESULTS_TXT_PATH", REMOVEWHOLE_PREDICTIONS_RESULTS_TXT_PATH)
    print("ROTATE_PREDICTIONS_TXT_PATH", ROTATE_PREDICTIONS_TXT_PATH)
    print("ROTATE_PREDICTIONS_RESULTS_TXT_PATH", ROTATE_PREDICTIONS_RESULTS_TXT_PATH)
    ########################TODO: NONE#########################
    print("NONE_PREDICTIONS_TXT_PATH", NONE_PREDICTIONS_TXT_PATH)
    print("NONE_PREDICTIONS_RESULTS_TXT_PATH", NONE_PREDICTIONS_RESULTS_TXT_PATH)
    print("termination_directory_list", termination_directory_list)
    print("NONE_TERMINATION_IMAGES", NONE_TERMINATION_IMAGES)

    if os.path.exists(directory_name):
        shutil.rmtree(directory_name)
    os.mkdir(directory_name)

    global g_debug_test_variable_info

    with open(TEST_VARIABLE_TXT_PATH, 'w') as f:
        f.write(g_debug_test_variable_info)
    return
##### multi batch friendly ####

def gather_filenames(termination_directories_list):
    """retrieve list of all termination filenames and corresponding labels
    
    
    """
    train_termination_files = []
    train_labels = []
    train_categories = []
    test_termination_files = []
    test_labels = []
    test_categories = []
    idx_count = 0
    for term_dir in termination_directories_list: 
        idx_count = 0
        for filename in glob.glob(term_dir+'*.png'): 
            object_category = check_object_category(filename)
            object_present_test = check_presence_test_object(object_category, term_dir)
            pair_label = find_label(term_dir)
            if object_present_test == False: #put in train set
                if object_category in train_object_categories_list:
                    train_termination_files.append(filename)
                    train_labels.append(pair_label)
                    train_categories.append(object_category)
                else: 
                    continue #not in train object categories
            else: # put in test set
                test_termination_files.append(filename)
                test_labels.append(pair_label)
                test_categories.append(object_category)

            idx_count += 1 
            if LIMIT_FLAG == True: 
                if idx_count > TEST_IMAGE_LIMIT:
                    break
    return train_termination_files, train_labels, train_categories, test_termination_files, test_labels, test_categories


def train_batch(train_filenames, train_labels, val_filenames, val_labels): 
    """Model and training
    
    Inputs: 
    - train_filenames: list of filenames for training images 
    - train_labels: list of labels that correspond to the training images
    - val_filenames: list of filenames for validation images
    - val_labels: list of labels that correspond to the validation images
    
    Returns: 
    None
    """
    # Generators
    training_generator = DataGenerator(train_filenames, train_labels, 'training_generator', SNAPSHOT_STEPS, class_dict=class_categories_dict,
                                       termination_directory_list=termination_directory_list,translate_initiation_directory=TRANSLATE_INITIATION_IMAGES,
                                       rotate_convert_labels=rotate_convert_labels, 
                                       batch_size=BATCH_SIZE, dim=(128,128), n_channels=3, shuffle=False, augmentation=False)

    validation_generator = DataGenerator(val_filenames, val_labels, 'validation_generator', SNAPSHOT_STEPS, class_dict=class_categories_dict, 
                                         termination_directory_list=termination_directory_list,translate_initiation_directory=TRANSLATE_INITIATION_IMAGES,
                                         rotate_convert_labels=rotate_convert_labels,
                                         batch_size=BATCH_SIZE, dim=(128,128), n_channels=3,
                                         shuffle=False, augmentation=False)
    
    batch_size = BATCH_SIZE
    print("batch_size", batch_size)

    input_init_train_keras = []
    for i in range(len(SNAPSHOT_STEPS)):
        img_train_one = np.zeros(shape=(batch_size, 128,128,3), dtype=np.uint8)
        input_train_one = tf.convert_to_tensor(img_train_one, dtype=tf.float32)
        input_init_train_keras_one = tf.keras.Input(shape=(128, 128), tensor = input_train_one)
        input_init_train_keras.append(input_init_train_keras_one)

    concate_input_train = tf.keras.layers.Concatenate()(input_init_train_keras)
    concate_input_train = tf.cast(concate_input_train, dtype=tf.float32)
    layer1 = Conv2D(32, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu', input_shape=(128,128,3))(concate_input_train)
    layer2 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(layer1)
    layer3 = Conv2D(64, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')(layer2)
    layer4 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(layer3)
    layer5 = Flatten()(layer4)
    layer6 = Dense(64, activation='relu')(layer5)
    layer7 = Dense(32, activation='relu')(layer6)
    probabilities = Dense(len(class_categories_dict), activation='softmax')(layer7)

    model = tf.keras.Model(inputs=input_init_train_keras, outputs=probabilities)    
    model.summary()
    lrr= ReduceLROnPlateau(monitor='val_accuracy', factor=.01, patience=3, min_lr=1e-5)
    batch_size= 32 #32
    print("batch_size:", batch_size)
    num_epochs=NUM_EPOCHS #50
    print("num_epochs:", num_epochs)
    learn_rate=.001

    adam_optimizer=Adam(learning_rate=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'], run_eagerly=True) 

    use_multiprocessing=False
    history = model.fit(training_generator, validation_data=validation_generator, callbacks=[lrr], verbose=1, use_multiprocessing=use_multiprocessing, epochs=num_epochs)

    # loss
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()
    plt.savefig(SAVED_LOSS_VAL_LOSS_PATH)
    print("saved LOSS_VAL_LOSS_PATH")
    plt.show(block=False)
    plt.pause(3)
    plt.close()

    # accuracies
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.legend()
    plt.savefig(SAVED_ACC_VAL_ACC_PATH)
    print("saved ACC_VAL_ACC")
    plt.show(block=False)
    plt.pause(3)
    plt.close()
    model.save(SAVED_MODEL_PATH)
    K.clear_session()   



def make_prediction_batch(test_termination_files, test_labels, generator_name, SAVED_MODEL_PATH): 
    """Testing preprocessing and prediction
    
    Inputs: 
    - test_termination_files: list, list of filepaths of images for testing
    - test_labels: list, list of labels that corresponds to the testing images
    - generator_name: str, name for the generator 
    - SAVED_MODEL_PATH: str, path for the model to predict with
    
    Returns: 
    - predictions: list, from the classifier given testing images
    """
    test_batch_size = BATCH_SIZE
    test_termination_files_size = len(test_termination_files)
    
    if test_termination_files_size < USE_BATCH_SIZE:
        test_batch_size = test_termination_files_size

    test_generator = DataGenerator(test_termination_files, test_labels, generator_name, SNAPSHOT_STEPS, class_dict=class_categories_dict, 
                                         termination_directory_list=termination_directory_list,translate_initiation_directory=TRANSLATE_INITIATION_IMAGES,
                                         rotate_convert_labels=rotate_convert_labels,
                                         batch_size=test_batch_size, dim=(128,128), n_channels=3,
                                         shuffle=False, augmentation=False)

    prediction = None        
    saved_model = load_model(SAVED_MODEL_PATH)

    if test_termination_files_size < USE_BATCH_SIZE:
        print("one batch")
        test_data = test_generator.get_items(0)
        prediction = saved_model(test_data[0], training=False)
    else:
        print("batches")
        prediction = saved_model.predict(test_generator)  

    print("prediction:", prediction)

    return prediction  

NO_AUGMENTATION = "no"
FLIP_AUGMENTATION = "flip"
ROTATE_AUGMENTATION = "rot"

def generate_augmentation_files(files_list, augment_list, rotate_angles):
    """function for producing augmentation
    
    Inputs:
    - files_list: list, list of files that need augmentation
    - augment_list: list, list of types of augmentation to do to files (No augmentation, Flip, Rotate)
    - rotate_angles: list, list of angles (in degrees) for rotation augmentation in 
    
    Returns: 
    - 2D list: (filename, type of augmentation)
    """
    twod_filelist = [(f, NO_AUGMENTATION) for f in files_list]
    if NO_AUGMENTATION in augment_list:
        return twod_filelist
    else:
        twod_flip_filelist = []
        twod_rotate_filelist = []
        for aug_type in augment_list:
            if aug_type == FLIP_AUGMENTATION:
                twod_flip_filelist = [(f, FLIP_AUGMENTATION) for f in files_list]
            if aug_type == ROTATE_AUGMENTATION:
                twod_rotate_filelist = []
                for rot_angle in rotate_angles:
                    twod_rotate_filelist = twod_rotate_filelist + [(f, ROTATE_AUGMENTATION+str(rot_angle)) for f in files_list]
        return twod_filelist + twod_flip_filelist + twod_rotate_filelist
                
def prepare_different_verbs(termination_folder, opposite_verb_label):
    """Grabs verbs, adds augmentation if needed. Also creates opposite verb sequences (ie: creates close
    from open)
    
    Inputs: 
    - termination_folder
    - opposite_verb_label: ie: if open verb, then close is opposite verb
    
    Returns: 
    Includes necessary info for train and testing sets (labels, categories, file paths) after
    augmentation and adding the opposite verbs files. 
    - open_close_train_termination_files
    - open_close_train_labels
    - open_close_train_categories
    - open_close_test_termination_files
    - open_close_test_labels
    - open_close_test_categories
    """
    open_train_termination_files, open_train_labels, open_train_categories, open_test_termination_files, open_test_labels, open_test_categories = gather_filenames([termination_folder])
    augment_list = []
    augment_count = 0
    if FLIP_FLAG == True:
        augment_list.append(FLIP_AUGMENTATION)
        augment_count +=1
    if ROTATE_FLAG == True:
        augment_list.append(ROTATE_AUGMENTATION)
        augment_count += len(ROTATION_AUGMENT)
    if len(augment_list) == 0:
        augment_list.append(NO_AUGMENTATION)
    open_train_termination_files = generate_augmentation_files(open_train_termination_files, augment_list, ROTATION_AUGMENT)
    open_test_termination_files = generate_augmentation_files(open_test_termination_files, [NO_AUGMENTATION], None)
    
    augment_open_labels = open_train_labels*augment_count 
    augment_open_categories = open_train_categories*augment_count
    open_train_labels = open_train_labels + augment_open_labels
    open_train_categories = open_train_categories + augment_open_categories
    
    open_close_train_termination_files = open_train_termination_files
    open_close_train_labels = open_train_labels
    open_close_train_categories = open_train_categories
    open_close_test_termination_files = open_test_termination_files
    open_close_test_labels = open_test_labels
    open_close_test_categories = open_test_categories
    #### 
    #opposite verb
    close_train_termination_files = []
    close_train_labels = []
    close_train_categories = []
    close_test_termination_files = []
    close_test_labels = []
    close_test_categories = []
    if not(opposite_verb_label == None):
        close_train_labels = [opposite_verb_label]*len(open_train_labels)
        close_test_labels = [opposite_verb_label]*len(open_test_labels)
        
        #added
        close_train_categories = open_train_categories.copy()
        close_test_categories = open_test_categories.copy()
        
        close_train_termination_files = []

        for terminal_train_file in open_train_termination_files:
            open_initiation_path = find_initiation_path(termination_folder, terminal_train_file[0])
            close_train_termination_files.append((open_initiation_path, terminal_train_file[1]))

        close_test_termination_files = []

        for terminal_test_file in open_test_termination_files:
            open_initiation_path = find_initiation_path(termination_folder, terminal_test_file[0])
            close_test_termination_files.append((open_initiation_path, terminal_test_file[1]))

        #train
        open_close_train_termination_files = open_close_train_termination_files + close_train_termination_files
        open_close_train_labels = open_close_train_labels + close_train_labels
        open_close_train_categories = open_close_train_categories + close_train_categories
        
        #test
        open_close_test_termination_files = open_close_test_termination_files + close_test_termination_files
        open_close_test_labels = open_close_test_labels +close_test_labels
        open_close_test_categories = open_close_test_categories + close_test_categories

    debug_and_print("----------prepare_different_verbs-----opposite_verb_label:{0}---".format(opposite_verb_label))
    debug_and_print("add files, termination_folder:{0}".format(termination_folder))
    debug_and_print("verb_train_termination_files size: {0}, verb_train_labels size: {1}, verb_train_categories size:{2}".format(len(open_train_termination_files), len(open_train_labels), len(open_train_categories)))
    debug_and_print("opposite_verb_train_termination_files size: {0}, opposite_verb_train_labels size: {1}, opposite_verb_train_categories size:{2}".format(len(close_train_termination_files), len(close_train_labels), len(close_train_categories)))
    debug_and_print("verb_test_termination_files size: {0}, verb_test_labels size: {1}, verb_test_categories:{2}".format(len(open_test_termination_files), len(open_test_labels), len(open_test_categories)))
    debug_and_print("opposite_verb_test_termination_files size: {0}, opposite_verb_test_labels size: {1}, opposite_verb_test_categories size:{2}".format(len(close_test_termination_files), len(close_test_labels), len(close_test_categories)))
    debug_and_print("verb_opposite_verb_test_termination_files size: {0}, verb_oppositeverb_test_labels size: {1}, verb_opposite_verb_test_categories size:{2}".format(len(open_close_test_termination_files), len(open_close_test_labels), len(open_close_test_categories)))
    
    return open_close_train_termination_files, open_close_train_labels, open_close_train_categories, open_close_test_termination_files, open_close_test_labels, open_close_test_categories

def counter_list_to_string(lst):
    str = ""
    c = collections.Counter(lst)
    index = 0
    total = 0
    for k,v in  c.most_common():
        str += "{0}){1}:{2}\n".format(index, k,v)
        total += v
        index += 1
    return str, total

def predictions_and_save_file(test_termination_files, test_labels, test_categories, generator_name, predictions_txt, predictions_results_txt):
    """After training, make predictions on unseen/test data. Write prediction results to file.  
    Inputs: 
    - test_termination_files
    - test_labels
    - test_categories
    - generator_name
    - predictions_txt
    - predictions_results_txt
    
    Returns: 
    None. Appends to file. 
    """
    # predict
    print("generator_name:", generator_name)
    print("test_termination_files len:", len(test_termination_files))
    print("test_labels len:", len(test_labels))
    predictions = make_prediction_batch(test_termination_files, test_labels, generator_name, SAVED_MODEL_PATH)
    # same result to two predict file
    correct_counter, incorrect_counter,correct_percentage = check_prediction(predictions, test_labels)

    np.savetxt(predictions_txt, predictions)
    with open(predictions_results_txt, 'w') as f:
        f.write("correct_counter:" + str(correct_counter) + "\n")
        f.write("incorrect_counter:" + str(incorrect_counter) + "\n")
        f.write("correct_percentage:" + str(correct_percentage) + "\n")

    # append result to log file
    counter_label_str, counter_label_total = counter_list_to_string(test_labels)
    counter_categories_str, ounter_categories_total = counter_list_to_string(test_categories)
    test_str = "-------prediction:{0}---------\n".format(generator_name)
    test_str += "count labels:{0}\n".format(counter_label_total)
    test_str += counter_label_str
    test_str += "count categories{0}:\n".format(ounter_categories_total)
    test_str += counter_categories_str
    test_str += "correct_counter:{0}\n".format(correct_counter)
    test_str += "incorrect_counter:{0}\n".format(incorrect_counter)
    test_str += "correct_percentage:{0}\n".format(correct_percentage)
    append_to_file(CLASSIFIER_EXE_LOG_TXT_PATH, test_str)

def append_train_info_to_log_file(train_labels, val_labels, train_categories, val_categories):
    """Model training info is documented in a log file."""
    str = "-------train info---------\n"
    counter_train_labels_str, counter_train_labels_total = counter_list_to_string(train_labels)
    str += "count_train_labels:{0}\n".format(counter_train_labels_total)
    str += counter_train_labels_str
    counter_train_categories_str, counter_train_categories_total = counter_list_to_string(train_categories)
    str += "count_train_categories:{0}\n".format(counter_train_categories_total)
    str += counter_train_categories_str
    counter_val_labels_str, counter_val_labels_total = counter_list_to_string(val_labels)
    str += "count_val_labels:{0}\n".format(counter_val_labels_total)
    str += counter_val_labels_str
    counter_val_categories_str, counter_val_categories_total = counter_list_to_string(val_categories)
    str += "count_val_categories:{0}\n".format(counter_val_categories_total)
    str += counter_val_categories_str
    append_to_file(CLASSIFIER_EXE_LOG_TXT_PATH, str)
        
        
def change_labels(labels_list):
    """For combining corresponding rotation CCW and CW into one verb. 
    ie: PitchCCW+PitchCW = Pitch verb
    Returns: 
    Labels_list after verbs are combined."""
    if not(ORIGINAL_ROLLCW == CONVERT_ROLLCW):
        labels_list = [CONVERT_ROLLCW if x == ORIGINAL_ROLLCW else x for x in labels_list]
    if not(ORIGINAL_ROLLCCW == CONVERT_ROLLCCW):
        labels_list = [CONVERT_ROLLCCW if x == ORIGINAL_ROLLCCW else x for x in labels_list]
    if not(ORIGINAL_PITCHCW == CONVERT_PITCHCW):
        labels_list = [CONVERT_PITCHCW if x == ORIGINAL_PITCHCW else x for x in labels_list]
    if not(ORIGINAL_PITCHCCW == CONVERT_PITCHCCW):
        labels_list = [CONVERT_PITCHCCW if x == ORIGINAL_PITCHCCW else x for x in labels_list]
    if not(ORIGINAL_YAWCW == CONVERT_YAWCW):
        labels_list = [CONVERT_YAWCW if x == ORIGINAL_YAWCW else x for x in labels_list]
    if not(ORIGINAL_YAWCCW == CONVERT_YAWCCW):
        labels_list = [CONVERT_YAWCCW if x == ORIGINAL_YAWCCW else x for x in labels_list]
    return labels_list

def main_batch():
    """logic for batching with data generators"""
    #gather filenames for terminations of train and tests, and their corresponding labels
    #TODO: need termination directories input
    open_close_test_termination_files = None
    open_close_test_labels = None
    open_close_test_categories = None
    
    removePart_insertPart_test_termination_files = None
    removePart_insertPart_test_labels = None
    removePart_insertPart_test_categories = None

    removeWhole_test_termination_files = None
    removeWhole_test_labels = None
    removeWhole_test_categories = None
    
    rotate_test_termination_files = None
    rotate_test_labels = None
    rotate_test_categories = None
    ########################none verb#########################
    none_train_termination_files = [None] * len(noneVerbDict)
    none_train_labels = [None] * len(noneVerbDict)
    none_train_categories = [None] * len(noneVerbDict)
    none_test_termination_files = [None] * len(noneVerbDict)
    none_test_labels = [None] * len(noneVerbDict)
    none_test_categories = [None] * len(noneVerbDict)
    ##########################################################
    
    train_termination_files, train_labels, train_categories, test_termination_files, test_labels, test_categories = gather_filenames(translate_termination_directory_list)
    train_termination_files = generate_augmentation_files(train_termination_files, [NO_AUGMENTATION], None)
    test_termination_files = generate_augmentation_files(test_termination_files, [NO_AUGMENTATION], None)
    
    translate_test_termination_files = test_termination_files
    translate_test_labels = test_labels
    translate_test_categories = test_categories
    
    
    debug_and_print("train_termination_files size: {0} train_labels size: {1}".format(len(train_termination_files), len(train_labels), len(train_categories)))
    debug_and_print("test_termination_files size: {0} test_labels size: {1}".format(len(test_termination_files), len(test_labels), len(test_categories)))
    
    if ROTATE_VERB_FLAG == True: 
        rotate_train_termination_files, rotate_train_labels, rotate_train_categories, rotate_test_termination_files, rotate_test_labels, rotate_test_categories = gather_filenames(rotate_termination_directory_list)
        rotate_train_termination_files = generate_augmentation_files(rotate_train_termination_files, [NO_AUGMENTATION], None)
        rotate_test_termination_files = generate_augmentation_files(rotate_test_termination_files, [NO_AUGMENTATION], None)

        rotate_train_labels = change_labels(rotate_train_labels)
        rotate_test_labels = change_labels(rotate_test_labels)
        
        train_termination_files = train_termination_files + rotate_train_termination_files
        train_labels = train_labels + rotate_train_labels
        train_categories = train_categories + rotate_train_categories
        
        test_termination_files = test_termination_files + rotate_test_termination_files
        test_labels = test_labels + rotate_test_labels
        test_categories = test_categories + rotate_test_categories
        debug_and_print("--------after rotate verb added---------")
        debug_and_print("train_termination_files size: {0} train_labels size: {1}".format(len(train_termination_files), len(train_labels), len(train_categories)))
        debug_and_print("test_termination_files size: {0} test_labels size: {1}".format(len(test_termination_files), len(test_labels), len(test_categories)))
        debug_and_print("----------------------------------------")


    if OPEN_FLAG == True:
        open_close_train_termination_files, open_close_train_labels, open_close_train_categories, open_close_test_termination_files, open_close_test_labels, open_close_test_categories = prepare_different_verbs(OPEN_TERMINATION_IMAGES, "close")
        train_termination_files = train_termination_files + open_close_train_termination_files
        train_labels = train_labels + open_close_train_labels
        train_categories = train_categories + open_close_train_categories
        
        test_termination_files = test_termination_files + open_close_test_termination_files
        test_labels = test_labels + open_close_test_labels
        test_categories = test_categories + open_close_test_categories
        

        debug_and_print("--------after open/close verb added---------")
        debug_and_print("train_termination_files size: {0}, train_labels size: {1}, train_categories size:{2}".format(len(train_termination_files), len(train_labels), len(train_categories)))
        debug_and_print("test_termination_files size: {0}, test_labels size: {1}, test_categories size:{2}".format(len(test_termination_files), len(test_labels), len(test_categories)))
    
    if REMOVEPART_FLAG == True:
        removePart_insertPart_train_termination_files, removePart_insertPart_train_labels, removePart_insertPart_train_categories, removePart_insertPart_test_termination_files, removePart_insertPart_test_labels, removePart_insertPart_test_categories = prepare_different_verbs(REMOVEPART_TERMINATION_IMAGES, "insertPart")
        train_termination_files = train_termination_files + removePart_insertPart_train_termination_files
        train_labels = train_labels + removePart_insertPart_train_labels
        train_categories = train_categories + removePart_insertPart_train_categories
        
        test_termination_files = test_termination_files + removePart_insertPart_test_termination_files
        test_labels = test_labels + removePart_insertPart_test_labels
        test_categories = test_categories + removePart_insertPart_test_categories
        debug_and_print("--------after remove part verbs added---------")
        debug_and_print("train_termination_files size: {0}, train_labels size: {1}, train_categories size:{2}".format(len(train_termination_files), len(train_labels), len(train_categories)))
        debug_and_print("test_termination_files size: {0}, test_labels size: {1}, test_categories size:{2}".format(len(test_termination_files), len(test_labels), len(test_categories)))

    if REMOVEWHOLE_FLAG == True:
        removeWhole_train_termination_files, removeWhole_train_labels, removeWhole_train_categories, removeWhole_test_termination_files, removeWhole_test_labels, removeWhole_test_categories = prepare_different_verbs(REMOVEWHOLE_TERMINATION_IMAGES, None)
        train_termination_files = train_termination_files + removeWhole_train_termination_files
        train_labels = train_labels + removeWhole_train_labels
        train_categories = train_categories + removeWhole_train_categories
        
        test_termination_files = test_termination_files + removeWhole_test_termination_files
        test_labels = test_labels + removeWhole_test_labels
        test_categories = test_categories + removeWhole_test_categories
        debug_and_print("--------after removeWhole verb added---------")
        debug_and_print("train_termination_files size: {0}, train_labels size: {1}, train_categories size:{2}".format(len(train_termination_files), len(train_labels), len(train_categories)))
        debug_and_print("test_termination_files size: {0}, test_labels size: {1}, test_categories size:{2}".format(len(test_termination_files), len(test_labels), len(test_categories)))

    ########################None verb#########################
    if NONE_VERB_FLAG == True:
        i = 0
        for k in noneVerbDict.keys():
            none_train_termination_files[i], none_train_labels[i], none_train_categories[i], none_test_termination_files[i], none_test_labels[i], none_test_categories[i] = prepare_different_verbs(NONE_TERMINATION_IMAGES[i], None)
            train_termination_files = train_termination_files + none_train_termination_files[i]
            train_labels = train_labels + none_train_labels[i]
            train_categories = train_categories + none_train_categories[i]
            
            test_termination_files = test_termination_files + none_test_termination_files[i]
            test_labels = test_labels + none_test_labels[i]
            test_categories = test_categories + none_test_categories[i]
            debug_and_print("--------after none verb -- {0} added---------".format(k))
            debug_and_print("train_termination_files size: {0}, train_labels size: {1}, train_categories size:{2}".format(len(train_termination_files), len(train_labels), len(train_categories)))
            debug_and_print("test_termination_files size: {0}, test_labels size: {1}, test_categories size:{2}".format(len(test_termination_files), len(test_labels), len(test_categories)))
            i += 1
    ########################################################

    
    #shuffle the termination filenames paired with their train labels (NOTE: not with shuffle_sets)
    train_termination_files_shuffled, train_labels_shuffled, train_categories_shuffled = shuffle(train_termination_files, train_labels, train_categories)
    
    # get validation filename and labels 
    train_filenames, train_labels, train_categories, val_filenames, val_labels, val_categories = train_validation_files(train_termination_files_shuffled, train_labels_shuffled, train_categories_shuffled, TRAIN_RATIO)
    
    debug_and_print("train_filenames size: {0}, train_labels size: {1}, train_categories size:{2}".format(len(train_filenames), len(train_labels), len(train_categories)))
    debug_and_print("val_filenames size: {0}, val_labels size: {1}, val_categories size:{2}".format(len(val_filenames), len(val_labels), len(val_categories)))

    counter_train_labels_str, counter_train_labels_total = counter_list_to_string(train_labels)
    debug_and_print("count_train_labels:{0}".format(counter_train_labels_total))
    debug_and_print(counter_train_labels_str)
    counter_val_labels_str, counter_val_labels_total = counter_list_to_string(val_labels)
    debug_and_print("count_val_labels:{0}".format(counter_val_labels_total))
    debug_and_print(counter_val_labels_str)
    counter_train_categories_str, counter_train_categories_total = counter_list_to_string(train_categories)
    debug_and_print("count_train_categories:{0}".format(counter_train_categories_total))
    debug_and_print(counter_train_categories_str)
    counter_val_categories_str, counter_val_categories_total = counter_list_to_string(val_categories)
    debug_and_print("count_val_categories:{0}".format(counter_val_categories_total))
    debug_and_print(counter_val_categories_str)
    counter_test_categories_str, counter_test_categories_total = counter_list_to_string(test_categories)
    debug_and_print("count_test_categories:{0}".format(counter_test_categories_total))
    debug_and_print(counter_test_categories_str)


    counter_translate_test_categories_str, counter_translate_test_categories_total = counter_list_to_string(translate_test_categories)
    debug_and_print("count_translate_test_categories:{0}".format(counter_translate_test_categories_total))
    debug_and_print(counter_translate_test_categories_str)
    
    if OPEN_FLAG == True: 
        counter_open_close_test_categories_str, counter_open_close_test_categories_total = counter_list_to_string(open_close_test_categories)
        debug_and_print("count_open_close_test_categories:{0}".format(counter_open_close_test_categories_total))
        debug_and_print(counter_open_close_test_categories_str)

    if REMOVEPART_FLAG == True:
        counter_removePart_insertPart_test_categories_str, counter_removePart_insertPart_test_categories_total = counter_list_to_string(removePart_insertPart_test_categories)
        debug_and_print("count_removePart_insertPart_test_categories:{0}".format(counter_removePart_insertPart_test_categories_total))
        debug_and_print(counter_removePart_insertPart_test_categories_str)

    if REMOVEWHOLE_FLAG == True:
        counter_removeWhole_test_categories_str, counter_removeWhole_test_categories_total = counter_list_to_string(removeWhole_test_categories)
        debug_and_print("count_removeWhole_test_categories:{0}".format(counter_removeWhole_test_categories_total))
        debug_and_print(counter_removeWhole_test_categories_str)

    if ROTATE_VERB_FLAG == True:
        counter_rotate_test_categories_str, counter_rotate_test_categories_total = counter_list_to_string(rotate_test_categories)
        debug_and_print("count_rotate_test_categories:{0}".format(counter_rotate_test_categories_total))
        debug_and_print(counter_rotate_test_categories_str)

    ########################none verb#########################
    if NONE_VERB_FLAG == True:
        i = 0
        for k in noneVerbDict.keys():
            counter_none_test_categories_str, counter_none_test_categories_total = counter_list_to_string(none_test_categories[i])
            debug_and_print("count_{0}_test_categories:{0}".format(k, counter_none_test_categories_total))
            debug_and_print(counter_none_test_categories_str)
            i += 1
    #########################################################

        
    with open(TEST_VARIABLE_TXT_PATH, 'a') as f:
        f.write(g_debug_test_variable_info)

    
    append_train_info_to_log_file(train_labels, val_labels, train_categories, val_categories)

    if LIMIT_FLAG  == True:
        train_filenames = train_filenames[0: CUT_OFF_TRAIN_COUNT]
        train_labels = train_labels[0: CUT_OFF_TRAIN_COUNT]
        train_categories = train_categories[0: CUT_OFF_TRAIN_COUNT]
        val_filenames = val_filenames[0: CUT_OFF_VAL_COUNT]
        val_labels = val_labels[0: CUT_OFF_VAL_COUNT]
        val_categories = val_categories[0: CUT_OFF_VAL_COUNT]
        debug_and_print("--------after cut of for test---------")
        debug_and_print("train_filenames size: {0}, train_labels size: {1}, train_categories size:{2}".format(len(train_filenames), len(train_labels), len(train_categories)))
        debug_and_print("val_filenames size: {0}, val_labels size: {1}, val_categories size:{2}".format(len(val_filenames), len(val_labels), len(val_categories)))
    
    train_batch(train_filenames, train_labels, val_filenames, val_labels)
    #overall
    predictions_and_save_file(test_termination_files, test_labels, test_categories, 'test_generator', PREDICTIONS_TXT_PATH, PREDICTIONS_RESULTS_TXT_PATH)
    #translate
    predictions_and_save_file(translate_test_termination_files, translate_test_labels, translate_test_categories, 'translate_test_generator', TRANSLATE_PREDICTIONS_TXT_PATH, TRANSLATE_PREDICTIONS_RESULTS_TXT_PATH)
    if OPEN_FLAG == True: 
        predictions_and_save_file(open_close_test_termination_files, open_close_test_labels, open_close_test_categories, 'open_close_test_generator', OPEN_CLOSE_PREDICTIONS_TXT_PATH, OPEN_CLOSE_PREDICTIONS_RESULTS_TXT_PATH)
    if REMOVEPART_FLAG == True:
        predictions_and_save_file(removePart_insertPart_test_termination_files, removePart_insertPart_test_labels, removePart_insertPart_test_categories, 'removePart_insertPart_test_generator', REMOVEPART_INSERTPART_PREDICTIONS_TXT_PATH, REMOVEPART_INSERTPART_PREDICTIONS_RESULTS_TXT_PATH)
    if REMOVEWHOLE_FLAG == True:
        predictions_and_save_file(removeWhole_test_termination_files, removeWhole_test_labels, removeWhole_test_categories, 'removeWhole_test_generator', REMOVEWHOLE_PREDICTIONS_TXT_PATH, REMOVEWHOLE_PREDICTIONS_RESULTS_TXT_PATH)
    if ROTATE_VERB_FLAG == True:
        predictions_and_save_file(rotate_test_termination_files, rotate_test_labels, rotate_test_categories, 'rotate_test_generator',ROTATE_PREDICTIONS_TXT_PATH,ROTATE_PREDICTIONS_RESULTS_TXT_PATH)
    ########################none verb#########################    
    if NONE_VERB_FLAG == True:
        i = 0
        for k in noneVerbDict.keys():
            predictions_and_save_file(none_test_termination_files[i], none_test_labels[i], none_test_categories[i], '{0}_test_generator'.format(k), NONE_PREDICTIONS_TXT_PATH[i], NONE_PREDICTIONS_RESULTS_TXT_PATH[i])
            i += 1

def append_to_file(fname, str):
    """append str to txt file
    
    Inputs: 
    - fname: str, path of file to append to
    - str: str to append to file
    
    Returns:
    None
    """
    with open(fname, 'a+') as f:
        f.write(str)

def main():
    datetime_start = datetime.now()
    start_time_str = datetime_start.strftime("%d-%b-%Y (%H:%M:%S.%f)")
    start_str = "=====>Start:{0}\n".format(start_time_str)
    append_to_file(CLASSIFIER_EXE_LOG_TXT_PATH, start_str)    

    read_command_line_arg()
    prepare_data()
    display_and_save_variable()
    
    execute_str = "train and test: {0}\n".format(directory_name)
    append_to_file(CLASSIFIER_EXE_LOG_TXT_PATH, execute_str)  

    main_batch()

    datetime_end = datetime.now()
    minutes_diff = (datetime_end - datetime_start).total_seconds() / 60.0
    end_time_str = datetime_end.strftime("%d-%b-%Y (%H:%M:%S.%f)")
    end_str = "<======End:{0}, total minutes:{1}\n".format(end_time_str, minutes_diff)
    append_to_file(CLASSIFIER_EXE_LOG_TXT_PATH, end_str)  



if __name__ == '__main__':

    main()