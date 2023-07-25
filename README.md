## Skill Generalization Via Verbs 
Classifier model and optimizer code for the paper: Skill Generalization Via Verbs.

## Setup 
See Requirements.txt for modules required.

The dataset created for the paper is available upon request. Please contact author at rma20@cs.brown.edu. 

## Command line arguments to run:

`python multistep_batch_classifier.py -EPOCHS 40 -OPEN_CLOSE True -REMOVE_PART_INSERT_PART True -REMOVE_WHOLE True -ROTATE_VERB True -NONE_VERB True -FLIP True -ROTATE False -Train_Categories_List Box,Dishwasher,Door,Laptop,Microwave,Oven,Refrigerator,Safe,Stapler,StorageFurniture,Toilet,TrashCan,WashingMachine -Test_Translate_Categories_List Box -Test_Open_Close_Categories_List Box -Test_RemovePart_InsertPart_Categories_List Box -Test_RemoveWhole_Categories_List Box -Test_Rotate_Categories_List Box -Test_None_Categories_List Box`




Command line arguments: 

To run training: 



