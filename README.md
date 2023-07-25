# Skill Generalization Via Verbs 
Classifier model and optimizer code for the paper: Skill Generalization Via Verbs.

## Setup 
See Requirements.txt for modules required.

The dataset created for the paper is available upon request. Please contact author at rma20@cs.brown.edu. 

# Command line arguments to run:

## Command line arguments for training classifier: 
 - `EPOCHS`: int number of epochs
 - `OPEN_CLOSE`, `-REMOVE_PART_INSERT_PART`, `-REMOVE_WHOLE`, `-ROTATE_VERB`, `-NONE_VERB`: true or false whether these verbs are to be included in the training+testing
 - `FLIP`: true or false whether to add flip augmentation to open/close verbs+remove_part/insert_part, none verbs for training
 - `ROTATE` true or false whether to add rotate augmentation  for training (was false for the experiments in paper)
- `Train_Categories_List`: commas (NO SPACES) separated of object categories to be trained on. It's ok to leave the anticipated test category, test categories will be removed from `Train_Categories_List`. 
- `Test_Translate_Categories_List`, `Test_Open_Close_Categories_List`,`Test_RemovePart_InsertPart_Categories_List`, `Test_RemoveWhole_Categories_List`, `-Test_Rotate_Categories_List`, `Test_None_Categories_List`: commas (NO SPACES) separated of object categories to be tested on. Test categories will be removed from `Train_Categories_List`. For experiments, same object category was entered for all of these. 

Below is an example of a training command for testing on the `Box` object category, 40 epochs, all verbs present: 
`python multistep_batch_classifier.py -EPOCHS 40 -OPEN_CLOSE True -REMOVE_PART_INSERT_PART True -REMOVE_WHOLE True -ROTATE_VERB True -NONE_VERB True -FLIP True -ROTATE False -Train_Categories_List Box,Dishwasher,Door,Laptop,Microwave,Oven,Refrigerator,Safe,Stapler,StorageFurniture,Toilet,TrashCan,WashingMachine -Test_Translate_Categories_List Box -Test_Open_Close_Categories_List Box -Test_RemovePart_InsertPart_Categories_List Box -Test_RemoveWhole_Categories_List Box -Test_Rotate_Categories_List Box -Test_None_Categories_List Box`





Command line arguments: 

To run training: 



