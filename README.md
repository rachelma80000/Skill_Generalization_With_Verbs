# Skill Generalization Via Verbs 
Classifier model and optimizer code for the accepted IROS paper: Skill Generalization Via Verbs.

Please make sure to cite this work if you are inspired or use the code: <br />

`@article{ma2023skill,` <br />
      `title={{Skill Generalization with Verbs}},`  <br />
      `author={Ma, Rachel and Lam, Lyndon and Spiegel, Benjamin A. and Ganeshan, Aditya and Patel,`  <br />
        `Roma and Abbatematteo, Ben and Paulius, David and Tellex, Stefanie and Konidaris, George},`  <br />
      `journal={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},`  <br />
      `year={2023},`  <br />
    `}` 

## Setup 
See Requirements.txt for modules required.

The images dataset created for the paper is available upon request. Please contact author at rma20@cs.brown.edu. 

To run code, you will need to edit `PROJECT_HOME_DIRECTORY` paths. It is also assumed that the `images/` folder is placed within `PROJECT_HOME_DIRECTORY`. Make sure files are unzipped. 

# Command line arguments to run:

## Command line arguments for training classifier: 
 - `EPOCHS`: int number of epochs
 - `OPEN_CLOSE`, `-REMOVE_PART_INSERT_PART`, `-REMOVE_WHOLE`, `-ROTATE_VERB`, `-NONE_VERB`: true or false whether these verbs are to be included in the training+testing
 - `FLIP`: true or false whether to add flip augmentation to open/close verbs+remove_part/insert_part, none verbs for training
 - `ROTATE` true or false whether to add rotate augmentation  for training (was false for the experiments in paper)
- `Train_Categories_List`: commas (NO SPACES) separated of object categories to be trained on. It's ok to leave the anticipated test category, test categories will be removed from `Train_Categories_List`. 
- `Test_Translate_Categories_List`, `Test_Open_Close_Categories_List`,`Test_RemovePart_InsertPart_Categories_List`, `Test_RemoveWhole_Categories_List`, `-Test_Rotate_Categories_List`, `Test_None_Categories_List`: commas (NO SPACES) separated of object categories to be tested on. Test categories will be removed from `Train_Categories_List`. For experiments, same object category was entered for all of these. 

Below is an example of a training command for testing on the `Box` object category, 40 epochs, all verbs present (single line, can be easily incorporated into .sh file):  
`python multistep_batch_classifier.py -EPOCHS 40 -OPEN_CLOSE True -REMOVE_PART_INSERT_PART True -REMOVE_WHOLE True -ROTATE_VERB True -NONE_VERB True -FLIP True -ROTATE False -Train_Categories_List Box,Dishwasher,Door,Laptop,Microwave,Oven,Refrigerator,Safe,Stapler,StorageFurniture,Toilet,TrashCan,WashingMachine -Test_Translate_Categories_List Box -Test_Open_Close_Categories_List Box -Test_RemovePart_InsertPart_Categories_List Box -Test_RemoveWhole_Categories_List Box -Test_Rotate_Categories_List Box -Test_None_Categories_List Box`

## Command line arguments for optimizer:
- `OBJECT_FOLDER_PATH`: path to folder containing URDF file of object instance to undergo manipulation
- `MODEL_DIRECTORY`: path to folder containing trained .h5 file. 
- `VERB`: target verb (this repository currently only contains code for optimizer manipulation for the following verbs: raise, lower, translateLeft, translateRight, push, pull, roll, turn, flip, open, close)

Below is ane example of training command for optimizer (single line, can be easily incorporated into .sh file): 

`python cma_es_trajectory.py -OBJECT_FOLDER_PATH box5/ -MODEL_DIRECTORY Classification_conv_40_flip-True_rotate-False_5steps_rotation_argment-304590_translateremoveWholerotatenoneBox_openremovePart-Box_removeWhole-Box -VERB raise`



