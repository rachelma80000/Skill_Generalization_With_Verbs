from re import S, U
from urdfpy import URDF
import json
from xml.dom import minidom
import os
import sys
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import os.path
import itertools
import argparse
import sapien.core as sapien
from PIL import Image, ImageColor
import open3d as o3d
from sapien.utils.viewer import Viewer
from transforms3d.euler import mat2euler
from cmaes import CMA
import math
import traceback
from datetime import datetime
import shutil
import time
import pandas as pd

import cma

DEBUG = False

PROJECT_HOME_DIRECTORY = "/home/rara/Documents/research_2022/demo_test/" #change to  home directory


PYTHON_SCRIPTS_DIRECTORY = PROJECT_HOME_DIRECTORY + "Goal-Transfer-Verbs/sapien-scripts/"

part_folder_path = PROJECT_HOME_DIRECTORY + "101940/"
test_urdf_file_path = part_folder_path + "mobility.urdf" 
test_meta_file_path = part_folder_path + "meta.json"   

EDITED_URDF_FILE = part_folder_path + "mab_edited_urdf.urdf"
INITIAL_IMG_DIRECTORY = part_folder_path + "initial_img.png"
MODEL_DIRECTORY = "Classification_conv_40_flip-True_rotate-False_5steps_rotation_argment-304590_translateremoveWholerotateSafe_openremovePart-Safe_removeWhole-Safe/"
SAVED_MODE_PATH = PROJECT_HOME_DIRECTORY+MODEL_DIRECTORY+"Classification_conv_40_flip-True_rotate-False_5steps_rotation_argment-304590_translateremoveWholerotateSafe_openremovePart-Safe_removeWhole-Safe.h5"

VERB = "raise"
cma_method = "cma-es"
penalty_rate1 = 1
penalty_rate2 = 1
#======SCALE========================
SCALE_UP_MATCH_CMA = 1 #scale up so that all delta can have the same mean and bounds for CMA-ES
SCALE_UP_MATCH_CMA_ROTATION = 4
#======BOUND========================
TRANSLATION_BOUND = 0.3
ROTATION_BOUND = 0.2
OPEN_CLOSE_BOUND = 0.2
#=====PYCMA LIMIT to avoid to run very long time========================
TIME_LIMIT_MINUTES = 10 #minutes
PYCMA_ITERATION_LIMIT = 100000000
#==============================
exe_name = part_folder_path + VERB + "_penalty1-"+str(penalty_rate1)+"/"

TEST_VARIABLE_TXT_PATH = exe_name+VERB + "_penalty1-"+str(penalty_rate1)+".txt"
DEBUG_FLAG = True
DEBUG_LOG_TXT_PATH = ""
LOG_TXT_PATH = PROJECT_HOME_DIRECTORY + "MAB_Log.txt"
g_debug_test_variable_info = ""
loss_numpy_file = exe_name+VERB + "_penalty1-"+str(penalty_rate1)+ ".npy"

from gpu_simple_prediction import main_without_parameters
from gpu_simple_prediction import main_without_parameters_SCALAR, retrieve_loss
#where the edited urdf files will be
EDITED_URDF_FILE_STEP_2_to_5 = [part_folder_path + "2.urdf", part_folder_path + "3.urdf", part_folder_path + "4.urdf", part_folder_path + "5.urdf"]
#where the rendered images will be
EDITED_IMG_DIRECTORY_STEP_2_to_5 = [part_folder_path + "mab_edited_img_2.png", part_folder_path + "mab_edited_img_3.png", part_folder_path + "mab_edited_img_4.png", part_folder_path + "mab_edited_img_5.png"]
MAB_DIR_ONE = part_folder_path+ "preview/"
MAB_DIR_TWO = MAB_DIR_ONE + "images_over_time_mini/"
MAB_PRODUCE_IMAGES_DIR = part_folder_path+"preview/images_over_time_mini/"+VERB+"_penalty-"+str(penalty_rate1)+"/"
LOSS_GENERATIONS_GRAPH = MAB_PRODUCE_IMAGES_DIR+ "loss_generation_graph.png"

camera_count = 0
CAMERA_RESET_CACHE_COUNT = 200
engine = None
renderer = None

current_gen = 0
current_pop = 0

DEGREE_AND_PENALTY = False

def get_original_max_limit(orig_urdf):
    """Get the original max and min joint limits. 
    Inputs: 
    - orig_urdf file
    Returns: 
    lower_limits, upper_limits"""
    lower_limits = []
    upper_limits = []

    file = minidom.parse(orig_urdf)
    joints = file.getElementsByTagName('joint')

    #number of joints w/ limits, or just limits, should be the same length as the list "units"
    limit_num = 0
    for joint in joints:
        if joint.getElementsByTagName('limit'):
            limit_line = joint.getElementsByTagName('limit')
            for lim in limit_line:
                max_upper_limit = lim.attributes['upper'].value     
                upper_limits.append(max_upper_limit)
                min_lower_limit = lim.attributes['lower'].value
                lower_limits.append(min_lower_limit)
                                    
                limit_num+=1
        else:
            #print("no joint limit")
            continue
    return lower_limits, upper_limits

def edit_limit_range_by_combo(orig_urdf, urdf_file, units, step): #edit all of the 'limit'
    """editing the joint limits (primarily for open/close verbs)"""
    lower_limits, upper_limits = get_original_max_limit(orig_urdf)

    file = minidom.parse(urdf_file)
    joints = file.getElementsByTagName('joint')
    updated = False

    #number of joints w/ limits, or just limits, should be the same length as the list "units"
    limit_num = 0
    for joint in joints:
        if joint.getElementsByTagName('limit'):
            limit_line = joint.getElementsByTagName('limit')
            for lim in limit_line:
                lower_limit_delta=units[limit_num] * SCALE_UP_MATCH_CMA_ROTATION #scale up so that all delta can have the same mean and bounds for CMA-ES
                max_upper_limit = upper_limits[limit_num]
                min_lower_limit = lower_limits[limit_num]
                lim.attributes['lower'].value = str(float(lim.attributes['lower'].value) + lower_limit_delta)
                if float(lim.attributes['lower'].value) > float(max_upper_limit): #if surpasses the upper limit
                    lim.attributes['lower'].value = max_upper_limit
                if float(lim.attributes['lower'].value) < float(min_lower_limit): #if surpasses the upper limit
                    lim.attributes['lower'].value = min_lower_limit
                lim.attributes['upper'].value = lim.attributes['lower'].value
                
                                       
                limit_num+=1
        else:
            #print("no joint limit")
            continue

    updated = True
    if updated == True:
        with open(urdf_file, "w") as fs:
            fs.write(file.toxml())
            fs.close()

def duplicate_orig_urdf(urdf_file, step):
    """make a copy of the original urdf"""
    file = minidom.parse(urdf_file)
    with open(EDITED_URDF_FILE_STEP_2_to_5[step], "w") as fs:  
        fs.write(file.toxml())
        fs.close()
            
def translate(urdf_file, units, step): #edits for both xyz and rpy; units is a list of len 6 [x y z r p y]
    """translate code"""
    updated = False
    file = minidom.parse(urdf_file)
    
    joints = file.getElementsByTagName('joint')
    
    for part in joints:
        type_joint = part.attributes['type'].value
        if type_joint == 'fixed':
            if part.getElementsByTagName('origin'):
                origin_line = part.getElementsByTagName('origin')
                for ori in origin_line: 
                    origin_coords_str = ori.attributes['xyz'].value
                    new_coords_str = manipulate_coords(origin_coords_str, [i * SCALE_UP_MATCH_CMA for i in units[0:3]]) #xyz 
                    ori.attributes['xyz'].value = new_coords_str
                    updated_coords = ori.attributes['xyz'].value
                    print("translation original:{0}, updated:{1}".format(origin_coords_str, str(updated_coords)))
            else:
                #print("no visual origin")
                continue
                
        else:
            continue
    updated = True
    if updated == True:
        with open(EDITED_URDF_FILE_STEP_2_to_5[step], "w") as fs:  
            fs.write(file.toxml())
            fs.close()
            
def rotate(urdf_file, units, step): #edits for both xyz and rpy; units is a list of len 6 [x y z r p y]
    """rotation code"""
    updated = False
    file = minidom.parse(urdf_file)
    joints = file.getElementsByTagName('joint')
    
    for part in joints:
        name = part.attributes['name'].value
        type_joint = part.attributes['type'].value
        if type_joint == 'fixed':
            if part.getElementsByTagName('origin'):
                origin_line = part.getElementsByTagName('origin')
                for ori in origin_line: 
                    origin_angs_str = ori.attributes['rpy'].value
                    new_coords_str = manipulate_coords(origin_angs_str, [i * SCALE_UP_MATCH_CMA_ROTATION for i in units[3:6]])
                    ori.attributes['rpy'].value = new_coords_str
                    updated_angs = ori.attributes['rpy'].value
                    print("rotation original:{0}, updated:{1}".format(origin_angs_str, str(updated_angs)))
                    
            else:
                continue    
        else:
            continue
    updated = True
    if updated == True:
        with open(EDITED_URDF_FILE_STEP_2_to_5[step], "w") as fs:  
            fs.write(file.toxml())
            fs.close()
    
    
def manipulate_coords(origin_coords_str, unit): #works for rpy too
    """Manipulating the coordinates for rotation or translation"""
    origin_coords = origin_coords_str.split(" ")
    try:
        x_coord = int(origin_coords[0])
    except:
        x_coord = float(origin_coords[0])
    try:
        y_coord = int(origin_coords[1])              
    except:
        y_coord = float(origin_coords[1])
    try:
        z_coord = int(origin_coords[2]) 
    except:
        z_coord = float(origin_coords[2])
        
    new_x_coord = x_coord
    new_y_coord = y_coord
    new_z_coord = z_coord

    new_x_coord = x_coord + unit[0]
    new_y_coord = y_coord + unit[1]
    new_z_coord = z_coord + unit[2]

    new_coords_arr = [str(new_x_coord), str(new_y_coord), str(new_z_coord)]
    new_coords_str = ' '.join(new_coords_arr)
    
    return new_coords_str


def render_loss(g, i): 
    """Opens edited urdf files in SAPIEN simulator and takes pictures; 
    pictures are sent to trained network to compute and return loss"""
    global camera_count
    global engine
    global renderer

    if camera_count == 0:
        engine = sapien.Engine()
        renderer = sapien.VulkanRenderer()
        engine.set_renderer(renderer)
    
    for step in range(len(EDITED_URDF_FILE_STEP_2_to_5)): 
        #take pictures of the 4 frames/steps after initiation image
        scene = engine.create_scene()
        scene.set_timestep(1 / 100.0)

        loader = scene.create_urdf_loader()
        loader.fix_root_link = True

        urdf_path = EDITED_URDF_FILE_STEP_2_to_5[step]
        # load as a kinematic articulation
        asset = loader.load_kinematic(urdf_path)
        assert asset, 'URDF not loaded.'

        rscene = scene.get_renderer_scene()
        rscene.set_ambient_light([0.5, 0.5, 0.5])
        rscene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        rscene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
        rscene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
        rscene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)
        # ---------------------------------------------------------------------------- #
        # Camera
        # ---------------------------------------------------------------------------- #
        near, far = 0.1, 100
        width, height = 128, 128
        camera_mount_actor = scene.create_actor_builder().build_kinematic()
        camera = scene.add_mounted_camera(
                name="camera",
                actor=camera_mount_actor,
                pose=sapien.Pose(),  # relative to the mounted actor
                width=width,
                height=height,
                fovx=np.deg2rad(35),
                fovy=np.deg2rad(35),
                near=near,
                far=far,
                )

        camera_coords=[-2, -2, 3]
                                                                                                                                                                                
        # Compute the camera pose by specifying forward(x), left(y) and up(z)
        cam_pos = np.array(camera_coords)
        forward = -cam_pos / np.linalg.norm(cam_pos)
        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left) #does not directly take pictures of top and bot views
        up = np.cross(forward, left)
        mat44 = np.eye(4)
        mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        mat44[:3, 3] = cam_pos
        camera_mount_actor.set_pose(sapien.Pose.from_transformation_matrix(mat44))
                                                                                  
        scene.step()  # make everything set
        scene.update_render()
        camera.take_picture()
        # ---------------------------------------------------------------------------- #
        # RGBA
        # ---------------------------------------------------------------------------- #
        rgba = camera.get_float_texture('Color')  # [H, W, 4]
        # An alias is also provided
        # rgba = camera.get_color_rgba()  # [H, W, 4]

########################################################################################

        rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
        rgba_pil = Image.fromarray(rgba_img)
        rgba_pil.save(EDITED_IMG_DIRECTORY_STEP_2_to_5[step]) #save images for NN
        rgba_pil.save(MAB_PRODUCE_IMAGES_DIR+str(g)+"_"+str(i)+"_"+str(step+2)+".png") #save for humans to view

    camera_count+=1
    if camera_count>=CAMERA_RESET_CACHE_COUNT:
        renderer.clear_cached_resources()
        camera_count=0
        time.sleep(5)

    edited_img = [INITIAL_IMG_DIRECTORY]+EDITED_IMG_DIRECTORY_STEP_2_to_5

    
    #using the scalar loss; cross entropy
    pred, target, reward = main_without_parameters_SCALAR(edited_img, SAVED_MODE_PATH, VERB)

########################################################################################

    return pred, target, reward


def build_guess_set(urdf_file):
    """build the guess parameters set"""
    params = np.array([0,0,0,0,0,0], dtype=float) #all urdf files have xyz and rpy by default
    file = minidom.parse(urdf_file)

    last_y=8 #last 'y' from rpy (yaw); always 5 for mini

    joints = file.getElementsByTagName('joint')
    for joint in joints:                                                   
        if joint.getElementsByTagName('limit'):
            limit_line = joint.getElementsByTagName('limit')
            for lim in limit_line:
                params = np.append(params, [0])
        else:
            #print("no joint limit")
            continue
    
    #3rd index is total number of limits
    return (params, last_y, params.size-1-last_y)


def loss_func(parameters):
    """Loss function for feedback from the neural network and adding to debug log."""
    probs, target, classifier_loss = render_loss(current_gen, current_pop)
    str = "prediction:{0}".format(probs)
    append_to_debug_file(DEBUG_LOG_TXT_PATH, str)    

    score = penalty_rate1*classifier_loss
    str = "Classifier_loss: {0}, Total_loss: {1}, penalty_rate1:{2}".format(classifier_loss, score, penalty_rate1 )
    print(str)
    append_to_debug_file(DEBUG_LOG_TXT_PATH, str)    
    return score

def to_urdf_parameter(parameters):
    """appending parameters"""
    x = parameters[0]
    y = parameters[1]
    z = parameters[2]
    urdf_parameters = [x, y, z]
    for i in range(3, len(parameters)):
        urdf_parameters.append(parameters[i])
    return urdf_parameters

def main_cma():
    """execute CMA_ES method"""
    params, last_y, num_limits = build_guess_set(test_urdf_file_path)
    print(params)

    cma_es_execution(params)

def debug(*args):
    if DEBUG == True:
        print(*args)

def cma_es_execution(params):
    """executing CMA-ES method"""
    global current_gen
    global current_pop
    bounds = None
    bounds = np.array([[-TRANSLATION_BOUND, TRANSLATION_BOUND],[-TRANSLATION_BOUND, TRANSLATION_BOUND],[-TRANSLATION_BOUND, TRANSLATION_BOUND], \
        [-ROTATION_BOUND, ROTATION_BOUND],[-ROTATION_BOUND, ROTATION_BOUND],[-ROTATION_BOUND, ROTATION_BOUND]])
    
    for i in range(params.size-6):
        bounds = np.append(bounds, [[-OPEN_CLOSE_BOUND,OPEN_CLOSE_BOUND]], 0)
    print("asked bounds length:", len(bounds))
    print("bounds:", bounds)

    pop_size = 40
    optimizer = CMA(mean=params, 
                    bounds=bounds, 
                    sigma=0.33,
                    n_max_resampling=1,
                    seed=0,
                    population_size = pop_size
                    )
    generations = 60

    str = "generations:{0},pop_size:{1}".format(generations, pop_size)
    append_to_debug_file(DEBUG_LOG_TXT_PATH, str)    
    
    #used to find the best parameters
    lowest_loss = 999999999
    best_parameters = 0

    avg_loss_across_generations = np.zeros(generations)
    loss_list = []
    step_gap = params.size #aka number of parameters for the urdf file
    for g in range(generations):
        total_loss = 0
        solutions = []

        for i in range(optimizer.population_size):
            print("--------------pupulation:{0}/{1}, generation:{2}/{3}-----------------".format(i,optimizer.population_size,g,generations))
            parameters_guess = optimizer.ask()
            ###take max attempt
            parameters_guess_max = np.zeros(params.size)
            index = np.argmax(np.absolute(parameters_guess))
            #max_value = np.amax(np.absolute(parameters_guess))
            actual_sign_value = parameters_guess[index]
            parameters_guess_max[index] = actual_sign_value
             
            print("asked paramaters_guess length:", len(parameters_guess))
            print("paramaters_guess:", parameters_guess)
            parameters = np.zeros(len(parameters_guess))
            if DEGREE_AND_PENALTY == True:
                for index in range(len(parameters_guess)):
                    if index >= 6:
                        parameters[index] = math.radians(parameters_guess[index])
                    else:
                        parameters[index] = parameters_guess[index]/100
            else:
                parameters = parameters_guess_max
            #parameters = optimizer.ask() #CMA-ES sample/guess paramters
            print("asked paramaters_guess length:", len(parameters_guess))
            str = "paramaters_guess:{0}".format(parameters_guess)            
            print(str)
            print("asked paramaters_max length:", len(parameters))
            str = "paramaters_max:{0}".format(parameters)            
            print(str)
            append_to_debug_file(DEBUG_LOG_TXT_PATH, str)    
            #four_params = np.concatenate((parameters, parameters, parameters, parameters), axis=None)
            urdf_parameters = parameters
            #urdf_parameters = to_urdf_parameter(parameters)
            print("urdf_parameters:", urdf_parameters)
    
            urdf_name = test_urdf_file_path
            for j in range(4):
                print("editing_urdf:", urdf_name)
                duplicate_orig_urdf(urdf_name, j)
                #test_key = input("enter test key:")
                #print("Test key is: " + test_key)
                translate(EDITED_URDF_FILE_STEP_2_to_5[j], list(urdf_parameters[0:6]), j)
                rotate(EDITED_URDF_FILE_STEP_2_to_5[j], list(urdf_parameters[0:6]), j)
                edit_limit_range_by_combo(test_urdf_file_path, EDITED_URDF_FILE_STEP_2_to_5[j], list(urdf_parameters[6:]), j)
                urdf_name = EDITED_URDF_FILE_STEP_2_to_5[j]
                #translate(test_urdf_file_path, list(parameters[0+j*step_gap:last_y+j*step_gap+1]), j)
                #rotate(EDITED_URDF_FILE_STEP_2_to_5[j], list(parameters[0+j*step_gap:last_y+j*step_gap+1]), j)
                #edit_limit_range_by_combo(test_urdf_file_path, EDITED_URDF_FILE_STEP_2_to_5[j], list(parameters[last_y+j*step_gap+1:last_y+num_limits+j*step_gap+1]))

            current_gen = g
            current_pop = i
            score = loss_func(parameters)

            total_loss+=score

            #loss1 = [classifier_loss, regulator_loss]
            solutions.append((parameters_guess, score))
            #loss_list.append(solutions)
            #find best scoring set
            current_loss = solutions[-1][1]
            if current_loss < lowest_loss:
                best_parameters = solutions[-1][0]
                #lowest_loss = solutions[-1][1]
                lowest_loss = current_loss
            
            #very first loss; used for reference
            if g==0 and i==0:
                init_score=score
            
            print("\n({0}, {1}/{2})_Score: {3} (Initial Score: {4}), Lowest Score: {5}".format(g, i+1, optimizer.population_size, score, init_score, lowest_loss))                


        avg_loss_across_generations[g] = total_loss/optimizer.population_size
        optimizer.tell(solutions) #update CMA-ES


    #np.savetxt(loss_numpy_file, solutions, delimiter=',')
    pd.DataFrame(solutions).to_csv(loss_numpy_file)
    plt.plot(avg_loss_across_generations)
    plt.xlabel("Generations")
    plt.ylabel("Average Loss")
    plt.savefig(LOSS_GENERATIONS_GRAPH)
    plt.title("Average Loss VS Generations")
    plt.show(block=False)
    plt.pause(3)
    plt.close()
    
    
    #visualize best scoring set
    #the 4 images will be labeled as 999_999_2.png, 999_999_3.png, 999_999_4.png, 999_999_5.png
    parameters_guess_max = np.zeros(params.size)
    index = np.argmax(np.absolute(best_parameters))
    #max_value = np.amax(np.absolute(parameters_guess))
    actual_sign_value = best_parameters[index]
    parameters_guess_max[index] = actual_sign_value
             
    parameters = parameters_guess_max
    urdf_parameters = parameters
    #urdf_parameters = to_urdf_parameter(parameters)
    
    urdf_name = test_urdf_file_path
    for j in range(4):
        print("editing_urdf:", urdf_name)
        duplicate_orig_urdf(urdf_name, j) 
        translate(EDITED_URDF_FILE_STEP_2_to_5[j], list(urdf_parameters[0:6]), j)
        rotate(EDITED_URDF_FILE_STEP_2_to_5[j], list(urdf_parameters[0:6]), j)
        edit_limit_range_by_combo(test_urdf_file_path, EDITED_URDF_FILE_STEP_2_to_5[j], list(urdf_parameters[6:]), j)
        urdf_name = EDITED_URDF_FILE_STEP_2_to_5[j]
    print(render_loss(999,999))
    print(avg_loss_across_generations)

    str = "best_parameters:{0}".format(best_parameters)
    append_to_debug_file(DEBUG_LOG_TXT_PATH, str)    

    
    
def debug_and_print(printStr):
    print(printStr)
    global g_debug_test_variable_info
    g_debug_test_variable_info += printStr + "\n"

def display_and_save_variable():
    print ("--------display variable ---------")
    debug_and_print("part_folder_path:{0}".format(part_folder_path))
    debug_and_print("test_urdf_file_path:{0}".format(test_urdf_file_path))
    debug_and_print("test_meta_file_path:{0}".format(test_meta_file_path))
    debug_and_print("EDITED_URDF_FILE:{0}".format(EDITED_URDF_FILE))
    debug_and_print("INITIAL_IMG_DIRECTORY:{0}".format(INITIAL_IMG_DIRECTORY))
    debug_and_print("SAVED_MODE_PATH:{0}".format(SAVED_MODE_PATH))
    debug_and_print("VERB:{0}".format(VERB))
    debug_and_print("cma_method:{0}".format(cma_method))
    debug_and_print("penalty_rate1:{0}".format(penalty_rate1))
    debug_and_print("penalty_rate2:{0}".format(penalty_rate2))
    debug_and_print("SCALE_UP_MATCH_CMA:{0}".format(SCALE_UP_MATCH_CMA))
    debug_and_print("SCALE_UP_MATCH_CMA_ROTATION:{0}".format(SCALE_UP_MATCH_CMA_ROTATION))
    debug_and_print("EDITED_URDF_FILE_STEP_2_to_5:{0}".format(EDITED_URDF_FILE_STEP_2_to_5))
    debug_and_print("EDITED_IMG_DIRECTORY_STEP_2_to_5:{0}".format(EDITED_IMG_DIRECTORY_STEP_2_to_5))
    debug_and_print("MAB_PRODUCE_IMAGES_DIR:{0}".format(MAB_PRODUCE_IMAGES_DIR))
    debug_and_print("LOSS_GENERATIONS_GRAPH:{0}".format(LOSS_GENERATIONS_GRAPH)) 
    debug_and_print("MAB_DIR_ONE:{0}".format(MAB_DIR_ONE))
    debug_and_print("MAB_DIR_TWO:{0}".format(MAB_DIR_TWO))       
    debug_and_print("exe_name:{0}".format(exe_name))    
    debug_and_print("TEST_VARIABLE_TXT_PATH:{0}".format(TEST_VARIABLE_TXT_PATH))
    debug_and_print("DEBUG_LOG_TXT_PATH:{0}".format(DEBUG_LOG_TXT_PATH))
    debug_and_print("LOG_TXT_PATH:{0}".format(LOG_TXT_PATH))
    debug_and_print("loss_numpy_file:{0}".format(loss_numpy_file))
    
    if not(os.path.exists(MAB_DIR_ONE)):
        os.mkdir(MAB_DIR_ONE)
    if not(os.path.exists(MAB_DIR_TWO)):
        os.mkdir(MAB_DIR_TWO)
    
    if os.path.exists(exe_name):
        shutil.rmtree(exe_name)
    os.mkdir(exe_name)
    
    if os.path.exists(MAB_PRODUCE_IMAGES_DIR):
        shutil.rmtree(MAB_PRODUCE_IMAGES_DIR)
    os.mkdir(MAB_PRODUCE_IMAGES_DIR)
    

    global g_debug_test_variable_info

    with open(TEST_VARIABLE_TXT_PATH, 'w') as f:
        f.write(g_debug_test_variable_info)
    return

def read_command_line_arg(): 
    """For arguments for bash file"""
    parser = argparse.ArgumentParser()
    try:
        parser.add_argument('-OBJECT_FOLDER_PATH', '--object_folder_path', type=str, help='path of folder not including project home', default="lab_cabinet/")
        parser.add_argument('-MODEL_DIRECTORY', '--model_directory', type=str, help='path of model', default="Classification_conv_80_flip-True_rotate-False_5steps_rotation_argment-304590_translateremoveWholerotatenoneStorageFurniture_openremovePart-StorageFurniture_removeWhole-StorageFurniture")
        parser.add_argument('-VERB', '--verb', type=str, help='verb command', default="raise") #raise open #roll, push
        parser.add_argument('-CMA', '--cma', type=str, help='cma type: cma-es', default="cma-es") 
        parser.add_argument('-penalty_rate1', '--penalty_rate1', type=float, help='penalty rate 1', default="1") 
        parser.add_argument('-penalty_rate2', '--penalty_rate2', type=float, help='penalty rate 2', default="10")
        args = parser.parse_args()
        print("--------input data ---------")
        print(f'OBJECT_FOLDER_PATH: {args.object_folder_path}')
        print(f'MODEL_DIRECTORY: {args.model_directory}')
        print(f'VERB: {args.verb}')
        print(f'CMA: {args.cma}')
        print(f'penalty_rate1: {args.penalty_rate1}')     
        print(f'penalty_rate2: {args.penalty_rate2}')     
        
        if not (args.cma.lower() == 'cma-es' or args.cma.lower() =='pycma'):
            raise 'only use cma-es or pycma'
            sys.exit(0)

        global part_folder_path
        global test_urdf_file_path
        global test_meta_file_path

        global EDITED_URDF_FILE
        global INITIAL_IMG_DIRECTORY
        global SAVED_MODE_PATH
        global VERB
        global cma_method
        global penalty_rate1
        global penalty_rate2

        global EDITED_URDF_FILE_STEP_2_to_5
        global EDITED_IMG_DIRECTORY_STEP_2_to_5
        global MAB_DIR_ONE
        global MAB_DIR_TWO
        global MAB_PRODUCE_IMAGES_DIR
        global LOSS_GENERATIONS_GRAPH
        
        global exe_name
        global TEST_VARIABLE_TXT_PATH
        global DEBUG_LOG_TXT_PATH
        global LOG_TXT_PATH
        global g_debug_test_variable_info
        global loss_numpy_file
        
        part_folder_path = PROJECT_HOME_DIRECTORY + args.object_folder_path
        test_urdf_file_path = part_folder_path + "mobility.urdf" 
        test_meta_file_path = part_folder_path + "meta.json"   

        EDITED_URDF_FILE = part_folder_path + "mab_edited_urdf.urdf"
        INITIAL_IMG_DIRECTORY = part_folder_path + "initial_img.png"
        model_directory = args.model_directory
        model_name = model_directory+".h5"
        print("model_name print:", model_name)
        SAVED_MODE_PATH = PROJECT_HOME_DIRECTORY+model_directory+"/"+model_name
        VERB = args.verb
        cma_method = args.cma
        penalty_rate1 = args.penalty_rate1
        penalty_rate2 = args.penalty_rate2
        
        name_with_verb_penalty = part_folder_path+VERB+"_penalty1-"+str(penalty_rate1)+"_penalty2-"+str(penalty_rate2)
        EDITED_URDF_FILE_STEP_2_to_5 = [name_with_verb_penalty + "_2.urdf", name_with_verb_penalty + "_3.urdf", name_with_verb_penalty + "_4.urdf", name_with_verb_penalty + "_5.urdf"]
        EDITED_IMG_DIRECTORY_STEP_2_to_5 = [name_with_verb_penalty + "_mab_edited_img_2.png", name_with_verb_penalty + "_mab_edited_img_3.png", name_with_verb_penalty + "_mab_edited_img_4.png", name_with_verb_penalty + "_mab_edited_img_5.png"]
        
        MAB_DIR_ONE = part_folder_path+ "preview/"
        MAB_DIR_TWO = MAB_DIR_ONE + "images_over_time_mini/"
        if not os.path.exists(MAB_DIR_ONE):
            os.makedirs(MAB_DIR_ONE)
        if not os.path.exists(MAB_DIR_TWO):
            os.makedirs(MAB_DIR_TWO)
        MAB_PRODUCE_IMAGES_DIR = part_folder_path+"preview/images_over_time_mini/"+VERB+"_penalty1-"+str(penalty_rate1)+"_penalty2-"+str(penalty_rate2)+"_"+datetime.now().strftime("%Y%m%d-%H%M")+"/"
        if not os.path.exists(MAB_PRODUCE_IMAGES_DIR):
            os.makedirs(MAB_PRODUCE_IMAGES_DIR)
        LOSS_GENERATIONS_GRAPH = MAB_PRODUCE_IMAGES_DIR+ "loss_generation_graph.png"
        
        exe_name = part_folder_path + VERB + "_penalty1-"+str(penalty_rate1) + "_penalty2-"+str(penalty_rate2)+"/"
        TEST_VARIABLE_TXT_PATH = exe_name+VERB + "_penalty1-"+str(penalty_rate1)+"_penalty2-"+str(penalty_rate2)+"_"+datetime.now().strftime("%Y%m%d-%H%M")+".txt"
        DEBUG_LOG_TXT_PATH = MAB_PRODUCE_IMAGES_DIR+"debug.txt"
        LOG_TXT_PATH = PROJECT_HOME_DIRECTORY + "MAB_Log.txt"
        loss_numpy_file = exe_name+VERB + "_penalty1-"+str(penalty_rate1)+"_penalty1-"+str(penalty_rate1)+"_"+datetime.now().strftime("%Y%m%d-%H%M")+".npy"
        
        g_debug_test_variable_info = ""
        
        return

    except Exception: 
        parser.print_help()
        traceback.print_exc()
        sys.exit(0)

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

def append_to_debug_file(fname, str):
    """append str to txt file
    
    Inputs: 
    - fname: str, path of file to append to
    - str: str to append to file
    
    Returns:
    None
    """
    if DEBUG_FLAG == False:
        return
    with open(fname, 'a+') as f:
        f.write(str+"\n")

        
def main():
    datetime_start = datetime.now()
    start_time_str = datetime_start.strftime("%d-%b-%Y (%H:%M:%S.%f)")
    start_str = "=====>Start:{0}\n".format(start_time_str)
    print(start_str)
    append_to_file(LOG_TXT_PATH, start_str)    

    read_command_line_arg()
    display_and_save_variable()
    
    execute_str = "optimizer: {0}\n".format(exe_name)
    append_to_file(LOG_TXT_PATH, execute_str)  
    main_cma()
    
    datetime_end = datetime.now()
    minutes_diff = (datetime_end - datetime_start).total_seconds() / 60.0
    end_time_str = datetime_end.strftime("%d-%b-%Y (%H:%M:%S.%f)")
    end_str = "<======End:{0}, total minutes:{1}\n".format(end_time_str, minutes_diff)
    append_to_file(LOG_TXT_PATH, end_str)  
    print(end_str)


if __name__ == '__main__':
    main()
    
 