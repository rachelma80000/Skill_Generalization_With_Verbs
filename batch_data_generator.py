
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np
import os
from keras import Sequential
from keras.utils import Sequence, to_categorical
import glob
from PIL import Image
import math
SHOW_DEBUG_INFO = False

NO_AUGMENTATION = "no"
FLIP_AUGMENTATION = "flip"
ROTATE_AUGMENTATION = "rot"

def debug(string):
    """debug function"""
    if SHOW_DEBUG_INFO == True:
        print(string)

class DataGenerator(Sequence):
    """multibatching of data for classifier training. """
    def __init__(self, list_filenames, labels, data_generator_name, snapshot_selection, class_dict, 
                termination_directory_list, translate_initiation_directory, rotate_convert_labels,
                batch_size, dim=(128,128), n_channels=3, shuffle=False, augmentation=False):
        'Initialization'
        self.OPEN_TERMINATION_IMAGES = termination_directory_list[0] #OPEN_INITIATION_IMAGES
        self.CLOSE_TERMINATION_IMAGES = termination_directory_list[1] #OPEN_TERMINATION_IMAGES #TRANSLATE_INITIATION_IMAGES
        self.REMOVEPART_TERMINATION_IMAGES = termination_directory_list[2]
        self.INSERTPART_TERMINATION_IMAGES = termination_directory_list[3]
        self.REMOVEWHOLE_TERMINATION_IMAGES = termination_directory_list[4]
        self.ROTATEROLLCW_TERMINATION_IMAGES = termination_directory_list[5]
        self.ROTATEROLLCCW_TERMINATION_IMAGES = termination_directory_list[6]
        self.ROTATEPITCHCW_TERMINATION_IMAGES = termination_directory_list[7]
        self.ROTATEPITCHCCW_TERMINATION_IMAGES = termination_directory_list[8]
        self.ROTATEYAWCW_TERMINATION_IMAGES = termination_directory_list[9]
        self.ROTATEYAWCCW_TERMINATION_IMAGES = termination_directory_list[10]
        self.NONE_TERMINATION_IMAGES_LIST = termination_directory_list[11:]
        self.TRANSLATE_INITIATION_IMAGES = translate_initiation_directory
        self.snapshot_selection = snapshot_selection
        self.class_dict = class_dict
        self.rotate_convert_labels = rotate_convert_labels
        #dimension is width and height
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.data_generator_name = data_generator_name
        self.list_filenames = list_filenames
        #n_channels is RGB: 3
        self.n_channels = n_channels
        #class of verbs
        self.n_classes = len(class_dict)
        #TODO: should be already done automatically?
        self.shuffle = shuffle
        #TODO: augmentation for open and close images only?
        self.augmentation = augmentation
        self.translate_labels = ["raise", "lower", "translateLeft", "translateRight", "push", "pull"]
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_filenames) / self.batch_size))

    def get_items(self, index):
        return self.__getitem__(index)
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        #name = self.data_generator_name
        #debug(f"__getitem__ -> {name}, index: {index}.")
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_filenames_temp = [self.list_filenames[k] for k in indexes]
        y_labels_temp = [self.labels[k] for k in indexes]
        # Generate data
        X, y_labels = self.__data_generation(list_filenames_temp, y_labels_temp)

        return X, y_labels

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def get_image(self,filename):
        """return arr of RGB image"""
        img = Image.open(filename)
        rgb = img.convert('RGB')
        img_arr = np.asarray(rgb)
        return img_arr
    
    def augment_flip_images(self,img_set):
        """Flip images augmentation math. Only apply to designated verbs."""
        flip_imgs = tf.image.flip_left_right(img_set)
        aug_data_set = flip_imgs.numpy().astype(np.uint8)
        return aug_data_set

    def augment_rotate_images(self, img_set, angle):
        """Rotate images augmentation math."""
        angle_rad = math.radians(angle)
        rotate_data = tfa.image.rotate(img_set, angle_rad)
        aug_data_set = rotate_data.numpy().astype(np.uint8)
        return aug_data_set
    
    def get_aug_image(self, img, augmentation_types):
        """Return designated augmentation for image"""
        if augmentation_types == NO_AUGMENTATION:  
            return img
        elif augmentation_types == FLIP_AUGMENTATION:
            return self.augment_flip_images(img)
        elif ROTATE_AUGMENTATION in augmentation_types: 
            angle = augmentation_types.replace(ROTATE_AUGMENTATION, "")
            return self.augment_rotate_images(img, float(angle))
        else:
            raise("invalid augmentation type")
    
    def get_converted_label(self, original_label):
        label = None
        for original, converted in self.rotate_convert_labels:
            if original_label == original:
                label = converted
                break
        return label

    def check_correct_label(self, label, original_label):
        converted_label = self.get_converted_label(original_label)
        if label == converted_label: 
            return True
        else:
            return False
    
    def find_snap_path(self, filename, snap_step, label):
        """get the filename of the snapshot given termination step filename"""
        snap_img_path = None
        term_dir_path = os.path.dirname(filename) + "/"
        fname = os.path.basename(filename)

        #print("fname:", fname) 
        if (term_dir_path == self.OPEN_TERMINATION_IMAGES and label == "open") or \
            (term_dir_path == self.REMOVEPART_TERMINATION_IMAGES and label == "removePart") or \
            (term_dir_path == self.REMOVEWHOLE_TERMINATION_IMAGES and label == "removeWhole") or \
            (term_dir_path == self.ROTATEROLLCW_TERMINATION_IMAGES and self.check_correct_label(label,"rotateRollCW")==True) or \
            (term_dir_path == self.ROTATEROLLCCW_TERMINATION_IMAGES and self.check_correct_label(label,"rotateRollCCW")==True) or \
            (term_dir_path == self.ROTATEPITCHCW_TERMINATION_IMAGES and self.check_correct_label(label,"rotatePitchCW")==True) or \
            (term_dir_path == self.ROTATEPITCHCCW_TERMINATION_IMAGES and self.check_correct_label(label,"rotatePitchCCW")==True) or \
            (term_dir_path == self.ROTATEYAWCW_TERMINATION_IMAGES and self.check_correct_label(label,"rotateYawCW")==True) or \
            (term_dir_path == self.ROTATEYAWCCW_TERMINATION_IMAGES and self.check_correct_label(label,"rotateYawCCW")==True) or \
            (term_dir_path in self.NONE_TERMINATION_IMAGES_LIST and label.startswith("none") == True):
            #it is for "open" label, and termination dir == OPEN_TERMINATION_IMAGES   
                   
            if snap_step in fname:
                # snap_step = termination 
                snap_img_path = filename
            elif snap_step == 'initiation':
                # snap_step = initiation, stepX 
                step_dir = term_dir_path.replace('termination', snap_step)
                step_filename = fname.replace('termination.png', snap_step+'.png')
                snap_img_path = step_dir + step_filename
            else:
                step_dir = term_dir_path.replace('_termination', "-"+snap_step)
                step_filename = fname.replace('termination.png', snap_step+'.png')
                snap_img_path = step_dir + step_filename
                
        elif (term_dir_path == self.CLOSE_TERMINATION_IMAGES and label == "close") or \
            (term_dir_path == self.INSERTPART_TERMINATION_IMAGES and label == "insertPart"):
            #it is for "close" lable, and termination dir == OPEN_INITIATION_IMAGES
            if snap_step in fname:
                # snap_step = initiation 
                snap_img_path = filename
            elif snap_step == 'termination': 
                # snap_step = termination, stepX 
                step_dir = term_dir_path.replace('initiation', snap_step)
                step_filename = fname.replace('initiation.png', snap_step+'.png')
                snap_img_path = step_dir + step_filename
            else:
                step_dir = term_dir_path.replace('_initiation', "-"+snap_step)
                step_filename = fname.replace('initiation.png', snap_step+'.png')
                snap_img_path = step_dir + step_filename
        elif (label in self.translate_labels):
            #if term_dir is one of the translate things
            step_dir = self.TRANSLATE_INITIATION_IMAGES
            parse_list = fname.split("-")
            if snap_step == "initiation":
                snap_img_path = step_dir + parse_list[0]+".png"
            elif snap_step == "termination":
                snap_img_path = filename
            else:
                step_dir_path = step_dir.replace("translate_initiation/", label+"/"+label+"-"+snap_step+"/")
                snap_img_path = step_dir_path + fname.replace(".png", "-" + snap_step+".png")
        else:
            snap_img_path = ""
        #debug("initiation_image_path:", initiation_image_path)
        return snap_img_path
    
    def __data_generation(self, list_filenames_temp, y_labels_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x_train = np.empty((len(self.snapshot_selection),self.batch_size, *self.dim, self.n_channels))
        #TODO: categorical
        Y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, termination_filename in enumerate(list_filenames_temp):
            ind_counter = 0
            label = y_labels_temp[i]
            snap_step_list = None
            if label == "close" or label == "insertPart": 
                # if it is close, it will do [open=termination, step15, step10, step5, close=initiation]
                snap_step_list = reversed(self.snapshot_selection)
            elif label == 'open':
                snap_step_list = self.snapshot_selection
            else:
                #it will do [initiation, step5, step10, step15, termination]
                snap_step_list = self.snapshot_selection
            for snap_step in snap_step_list:
                snap_filename = self.find_snap_path(termination_filename[0], snap_step, label)
                img = self.get_image(snap_filename)    
                #debug(f"__data_generation: snap_filename:{snap_filename}")
                x_train[ind_counter, i,] = self.get_aug_image(img, termination_filename[1])
                ind_counter += 1
                
        Y = [self.class_dict[k] for k in y_labels_temp]
        X = []
        # X is array of multi input, for examples, 
        for i in range(len(self.snapshot_selection)):
            X.append(x_train[i])

        return X, to_categorical(Y, num_classes=self.n_classes, dtype = 'float32')