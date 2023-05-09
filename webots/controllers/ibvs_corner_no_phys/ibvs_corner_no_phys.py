#  ...........       ____  _ __
#  |  ,-^-,  |      / __ )(_) /_______________ _____  ___
#  | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  | / ,..´  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#     +.......   /_____/_/\__/\___/_/   \__,_/ /___/\___/
 
# MIT License

# Copyright (c) 2022 Bitcraze

# @file crazyflie_controllers_py.py
# Controls the crazyflie motors in webots in Python

"""crazyflie_controller_py controller."""


# from controller import Robot
from controller import Supervisor
from controller import Motor
from controller import InertialUnit
from controller import GPS
from controller import Gyro
from controller import Keyboard
from controller import Camera
from controller import DistanceSensor

import os, datetime
import cv2
import csv, json
import numpy as np
import random
import errno
import itertools
import transforms3d
import machinevisiontoolbox as mvtb
from spatialmath import SE3

random.seed()

from math import cos, sin, degrees, radians
from ai import cs
import matplotlib.pyplot as plt

import sys
sys.path.append('../../../controllers/')

# Import the path for the corner detection module
sys.path.append('../../../../scarciglia-nanodrone-gate-detection/')

from src import corner, rotation, geometry

# Change this path to your crazyflie-firmware folder
sys.path.append('../../../../crazyflie-firmware/build')

import cffirmware

##### My ADDITION #####
# print(sys.path)
# print(os.getcwd())
# print(os.listdir('../../../controllers/'))
##### My ADDITION #####
from  pid_controller import init_pid_attitude_fixed_height_controller, pid_velocity_fixed_height_controller
from pid_controller import MotorPower_t, ActualState_t, GainsPID_t, DesiredState_t
robot = Supervisor()

timestep = int(robot.getBasicTimeStep()) # original

## Initialize motors
m1_motor = robot.getDevice("m1_motor");
m1_motor.setPosition(float('inf'))
m1_motor.setVelocity(-1)
m2_motor = robot.getDevice("m2_motor");
m2_motor.setPosition(float('inf'))
m2_motor.setVelocity(1)
m3_motor = robot.getDevice("m3_motor");
m3_motor.setPosition(float('inf'))
m3_motor.setVelocity(-1)
m4_motor = robot.getDevice("m4_motor");
m4_motor.setPosition(float('inf'))
m4_motor.setVelocity(1)

## Initialize Sensors
imu = robot.getDevice("inertial unit")
imu.enable(timestep)
gps = robot.getDevice("gps")
gps.enable(timestep)
gyro = robot.getDevice("gyro")
gyro.enable(timestep)
camera = robot.getDevice("camera")
camera.enable(timestep)

####
# gps_camera = robot.getDevice("gps_camera")
# gps_camera.enable(timestep)
# imu_camera = robot.getDevice("imu_camera")
# imu_camera.enable(timestep)
####

range_front = robot.getDevice("range_front")
range_front.enable(timestep)
range_left = robot.getDevice("range_left")
range_left.enable(timestep)
range_back = robot.getDevice("range_back")
range_back.enable(timestep)
range_right = robot.getDevice("range_right")
range_right.enable(timestep)

# Crazyflie
crazyflie_node = robot.getFromDef("CRAZYFLIE")
translation_drone = crazyflie_node.getField('translation')
rotation_drone = crazyflie_node.getField('rotation')
camera_node = crazyflie_node.getField('children').getMFNode(1)
camera_drone_tr = camera_node.getField('translation').getSFVec3f()

# Gate
gate_node = robot.getFromDef("GATE")
translation_gate = gate_node.getField('translation').getSFVec3f()
gate_center = gate_node.getField('children').getMFNode(4).getField('translation').getSFVec3f()
br = gate_node.getField('children').getMFNode(0).getField('translation').getSFVec3f()
bl = gate_node.getField('children').getMFNode(1).getField('translation').getSFVec3f()
tl = gate_node.getField('children').getMFNode(2).getField('translation').getSFVec3f()
tr = gate_node.getField('children').getMFNode(3).getField('translation').getSFVec3f()
P = np.array([
    tl, bl, br, tr
]).T
print('Gate data:', br, bl, tl, tr, gate_center, translation_gate)

## Get keyboard
keyboard = Keyboard()
keyboard.enable(timestep)

cffirmware.controllerPidInit()

print('Take off!')

## Initialize variables
pastXGlobal = 0
pastYGlobal = 0
pastZGlobal = 0
pastRoll = 0
pastPitch = 0
pastYaw = 0

past_time = robot.getTime()

cffirmware.controllerPidInit()

########### ------------------ VISUAL SERVOING ------------------ ###########
f = 0.0006
pixel_size = (3.6e-6, 3.6e-6)
img_size = (320,320)
cam = mvtb.CentralCamera(rho=pixel_size[0], imagesize=img_size, f=f)

# Depth value
Z = 0.34

# Desired positions
wide = 200 # square 400px wide in the center of the camera frame
pd = np.array([[cam.pp[0] - wide/2, cam.pp[1] - wide/2], # TL is the 1st
               [cam.pp[0] - wide/2, cam.pp[1] + wide/2], # BL is the 2nd
               [cam.pp[0] + wide/2, cam.pp[1] + wide/2], # BR is the 3rd
               [cam.pp[0] + wide/2, cam.pp[1] - wide/2]])# TR is the 4th
pd = pd.T # set the x and y coordinates by column

lmda = 0.08

thresh = 5
err = np.inf

old_p_detected = None

########### ------------------ VISUAL SERVOING ------------------ ###########

########### ------------------ SAVING THINGS -------------------- ###########
import pickle

# Set to True if you want to collect data
collect_data = False

dataset_path = '../../datasets/EXP-2-FRAME-IBVS-INTEGRATION/CORNER'
imgs_folder = f'{dataset_path}/imgs/'
imgs_ibvs_folder = f'{dataset_path}/imgs_ibvs/'

try:
    os.makedirs(imgs_folder)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise  # This was not a "directory exist" error..
    else:
        if collect_data:
            print(f"Overwriting folder {imgs_folder}")

try:
    os.makedirs(imgs_ibvs_folder)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise  # This was not a "directory exist" error..
    else:
        if collect_data:
            print(f"Overwriting folder {imgs_ibvs_folder}")

dataset = {}
dataset['info'] = {'description': "Dataset used to test the IBVS pipeline. Consider the drone as a frame, thus no physics is involved. It uses the angular velocity of the drone [roll, pitch, and yaw] (twist camera-drone) and the X, Y, and Z position of the world frame (twist camera-world). Velocities are v_x, v_y, v_z, w_x, w_y, w_z (yaw rate). The next position and orientation is computed by integrating linear and angular velocities.",
                   'used_corner_detection':True}
dataset['drone'] = {'starting_position': crazyflie_node.getField('translation').getSFVec3f(),
                    'starting_rotation': crazyflie_node.getField('rotation').getSFRotation(),
                    'camera_drone_tr': camera_node.getField('translation').getSFVec3f()}
dataset['gate'] = {'corners':{'tl':tl, 'bl':bl, 'br':br, 'tr':tr},
                   'position': gate_node.getField('translation').getSFVec3f(),
                   'rotation': gate_node.getField('rotation').getSFRotation()}
dataset['camera'] = {'f':f, 'pixel_size':pixel_size, 'img_size':img_size }
dataset['ibvs'] = {'lambda': lmda, 'threshold': thresh, 'Z':{'estimated':False, 'value':Z}}

samples = {}
frame_n = 0
########### ------------------ SAVING THINGS -------------------- ###########

# OpenCV show images
cv2.startWindowThread()
cv2.namedWindow("preview")
colors = ['r', 'b', 'g', 'y']

idx = 0

while robot.step(timestep) != -1 and err > thresh: # and frame_n < n_samples: # with this condition the controller will exit

    dt = robot.getTime() - past_time
    
    ## Get measurements
    roll, pitch, yaw = imu.getRollPitchYaw()
    roll_rate, pitch_rate, yaw_rate = gyro.getValues()
    xGlobal, yGlobal, zGlobal = gps.getValues()
    vxGlobal = (xGlobal - pastXGlobal)/dt
    vyGlobal = (yGlobal - pastYGlobal)/dt
    vzGlobal = (zGlobal - pastZGlobal)/dt
    
    # CAMERA IMAGES
    w, h = camera.getWidth(), camera.getHeight()
    cameraData = camera.getImage()  # Note: uint8 string
    image = np.frombuffer(cameraData, np.uint8).reshape(h, w, 4) # BGR, alpha (transparency)
    
    ########### ------------------ ROTATIONS ------------------ ###########
    rotation_matrix_world_drone = rotation.rotation_matrix(roll, pitch, yaw)
    wd_tr = np.array([xGlobal, yGlobal, zGlobal])
    extrinsic_matrix_world_drone = rotation.get_extrinsic_matrix(rotation_matrix_world_drone, translation_vector=wd_tr)
    
    rotation_matrix_drone_camera = rotation.rotation_matrix(roll=-np.pi/2, pitch=0, yaw=-np.pi/2)
    dc_tr = camera_drone_tr
    extrinsic_matrix_drone_camera = rotation.get_extrinsic_matrix(rotation_matrix_drone_camera, translation_vector=dc_tr)

    extrinsic_matrix = extrinsic_matrix_world_drone@extrinsic_matrix_drone_camera
    ########### ------------------ ROTATIONS ------------------ ###########
    
    
    ########### ------------------ CORNER DETECTION ------------------ ###########
        
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    p_detected = corner.detect_corners(img)
    
    # if idx == 0:
    #     p_detected = corner.detect_corners(img)
    # else:
    #     current_p_detected = corner.detect_corners(img)
    #     if current_p_detected is None:
    #         p_detected = old_p_detected
    #     else:
    #         p_detected = corner.weigh_detection(current_p_detected, old_p_detected, alpha=.2)
    
    old_p_detected = p_detected
    
    ########### ------------------ CORNER DETECTION ------------------ ###########

    ########### ------------------ VISUAL SERVOING ------------------ ###########
    # image-plane error
    try:
        e = pd - p_detected
        err = np.linalg.norm(e)
    except:
        continue
    
    # stacked image Jacobian
    J = cam.visjac_p(p_detected, Z) # Z = 0.34
    try:
        v_camera = lmda * np.linalg.pinv(J) @ e.T.flatten()
    except Exception as e:
        print(e)
        break
        
    # Twist velocity from camera frame to world frame
    rot = np.array(extrinsic_matrix)[:3,:3] # from World to Camera
    t = np.array(extrinsic_matrix)[:3,-1] # translation vector world camera
    # Since we move the drone in the world-coordinate, we need a twist matrix between the world and the camera
    twist_world_camera = geometry.velocity_twist_matrix(extrinsic_matrix[:3,:3], extrinsic_matrix[:3,-1])
    v_world = twist_world_camera@v_camera
    v_x, v_y, v_z, w_x, w_y, w_z = v_world
    
    twist_drone_camera = geometry.velocity_twist_matrix(rotation_matrix_drone_camera, dc_tr)
    v_drone = twist_drone_camera@v_camera
    v_x_drone, v_y_drone, v_z_drone, w_x_drone, w_y_drone, w_z_drone = v_drone
    
    ########### ------------------ VISUAL SERVOING ------------------ ###########
    
    ########### ------------------ INTEGRATION ------------------ ###########
    
    past_time = robot.getTime()
    
    pastXGlobal = xGlobal
    pastYGlobal = yGlobal
    pastZGlobal = zGlobal
    
    pastRoll = roll
    pastPitch = pitch
    pastYaw = yaw
    nextX = pastXGlobal + v_x*dt
    nextY = pastYGlobal + v_y*dt
    nextZ = pastZGlobal + v_z*dt
     
    nextRoll = pastRoll + w_x_drone*dt
    nextPitch = pastPitch + w_y_drone*dt
    nextYaw = pastYaw + w_z_drone*dt 
    
    # nextRoll = pastRoll + w_x*dt
    # nextPitch = pastPitch + w_y*dt
    # nextYaw = pastYaw + w_z*dt
     
    next_rot = rotation.rotation_matrix(nextRoll, nextPitch, nextYaw)
    rot_ax = transforms3d.axangles.mat2axangle(next_rot)
    ax_angle = [rot_ax[0][0], rot_ax[0][1], rot_ax[0][2]] # axis
    ax_angle.append(rot_ax[-1]) # append the angle
    
    ########### ------------------ INTEGRATION ------------------ ###########
    
    # Set the pose
    translation_drone.setSFVec3f([nextX, nextY, nextZ])    
    crazyflie_node.getField('rotation').setSFRotation(ax_angle)
    
    # print(f'vx:{v_x:.4f}\tvy:{v_y:.4f}\tvz:{v_z:.4f}')
    # print(f'wx:{w_x:.4f}\twy:{w_y:.4f}\twz:{w_z:.4f}')
    print(f'vx_drone:{v_x_drone:.4f}\tvy_drone:{v_y_drone:.4f}\tvz_drone:{v_z_drone:.4f}')
    print(f'wx_drone:{w_x_drone:.4f}\twy_drone:{w_y_drone:.4f}\twz_drone:{w_z_drone:.4f}')
    print(f'error:{np.linalg.norm(e):.4f}')
    
    # Increase the index
    idx += 1
    
    # Show image and save the corresponding images
    ibvs_img_path = f"{imgs_ibvs_folder}/img_{frame_n}.png"
    for id, col in enumerate(colors):
        tl = pd[:,0]
        bl = pd[:,1]
        br = pd[:,2]
        tr = pd[:,-1]
        x, y = pd[:,id]
        image = cv2.circle(image, (int(x),int(y)), radius=2, color=(255, 255, 255), thickness=1)
        image = cv2.line(image, (int(tl[0]), int(tl[1])), (int(tr[0]), int(tr[1])), color=(255, 255, 255), thickness=1) # top-left, top-right
        image = cv2.line(image, (int(tr[0]), int(tr[1])), (int(br[0]), int(br[1])), color=(255, 255, 255), thickness=1) # top-right, bottom-right
        image = cv2.line(image, (int(br[0]), int(br[1])), (int(bl[0]), int(bl[1])), color=(255, 255, 255), thickness=1) # bottom-left, top-right
        image = cv2.line(image, (int(bl[0]), int(bl[1])), (int(tl[0]), int(tl[1])), color=(255, 255, 255), thickness=1) # bottom-left, top-left
        x, y = p_detected[:,id]
        image = cv2.circle(image, (int(x),int(y)), radius=2, color=(255, 255, 255), thickness=-1)
                
        if collect_data:
            # Save the image
            cv2.imwrite(ibvs_img_path, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    
    cv2.imshow("preview", cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    cv2.waitKey(timestep)
    
    ########### ------------------ SAVING THINGS -------------------- ########### 

    if collect_data:
        sample = {}
        sample['global_position'] = [xGlobal, yGlobal, zGlobal]
        sample['global_velocities'] = [vxGlobal, vyGlobal, vzGlobal, roll_rate, pitch_rate, yaw_rate]
        sample['ibvs_velocities_body_frame'] = [v_x_drone, v_y_drone, v_z_drone, w_x_drone, w_y_drone, w_z_drone]
        sample['ibvs_velocities_world_frame'] = [v_x, v_y, v_z, w_x, w_y, w_z]
        sample['target_points'] = pd
        sample['detected_points'] = p_detected
        sample['ibvs_error'] = err
        
        samples[f"img_{frame_n}.png"] = sample
        
        # Save images
        path = f"{imgs_folder}/img_{frame_n}.png"
        cv2.imwrite(path, img) # gray-scale image
        frame_n += 1
        
    ########### ------------------ SAVING THINGS -------------------- ###########

    
dataset['samples'] = samples

########### ------------------ SAVING THINGS -------------------- ###########
if collect_data:
    print("Saving data...")
    with open(dataset_path + '/data.pickle', 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Data saved.")
########### ------------------ SAVING THINGS -------------------- ###########