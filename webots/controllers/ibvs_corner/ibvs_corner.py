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
random.seed()

from math import cos, sin, degrees, radians
from ai import cs

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

thresh = 5e-1
err = np.inf

########### ------------------ VISUAL SERVOING ------------------ ###########

start_vs = False

while robot.step(timestep) != -1 and err > thresh: # and frame_n < n_samples: # with this condition the controller will exit

    dt = robot.getTime() - past_time;
    
    ## Get measurements
    roll, pitch, yaw = imu.getRollPitchYaw()
    roll_rate, pitch_rate, yaw_rate = gyro.getValues()
    xGlobal, yGlobal, zGlobal = gps.getValues()
    vxGlobal = (xGlobal - pastXGlobal)/dt
    vyGlobal = (yGlobal - pastYGlobal)/dt
    vzGlobal = (zGlobal - pastZGlobal)/dt

    
    ## Put measurement in state estimate
    # TODO replace these with a EKF python binding
    state = cffirmware.state_t()
    state.attitude.roll = degrees(roll)
    state.attitude.pitch = -degrees(pitch)
    state.attitude.yaw = degrees(yaw) # - starting_yaw
    state.position.x = xGlobal
    state.position.y = yGlobal
    state.position.z = zGlobal
    state.velocity.x = vxGlobal
    state.velocity.y = vyGlobal
    state.velocity.z = vzGlobal
    
    # Put gyro in sensor data
    sensors = cffirmware.sensorData_t()
    sensors.gyro.x = degrees(roll_rate)
    sensors.gyro.y = degrees(pitch_rate)
    sensors.gyro.z = degrees(yaw_rate)
    
    # CAMERA IMAGES
    w, h = camera.getWidth(), camera.getHeight()
    cameraData = camera.getImage()  # Note: uint8 string
    image = np.frombuffer(cameraData, np.uint8).reshape(h, w, 4) # BGR, alpha (transparency)
    
    if start_vs:
        ########### ------------------ CONTOUR PIPELINE ------------------ ###########
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # print(type(img))
        # print(os.getcwd())
        p_detected = corner.detect_corners(img)
        print(p_detected)
        ########### ------------------ CONTOUR PIPELINE ------------------ ###########
        
        ########### ------------------ VISUAL SERVOING ------------------ ###########
        # image-plane error
        try:
            e = pd - p_detected
        except:
            continue
        
        # stacked image Jacobian
        J = cam.visjac_p(p_detected, Z) # Z = 0.34
        v_camera = lmda * np.linalg.pinv(J) @ e.T.flatten()
        
                    ########### ------------------ ROTATIONS ------------------ ###########
        rotation_matrix_world_drone = rotation.rotation_matrix(roll, pitch, yaw)
        wd_tr = np.array([xGlobal, yGlobal, zGlobal])
        extrinsic_matrix_world_drone = rotation.get_extrinsic_matrix(rotation_matrix_world_drone, translation_vector=wd_tr)
        
        rotation_matrix_drone_camera = rotation.rotation_matrix(roll=-np.pi/2, pitch=0, yaw=-np.pi/2)
        dc_tr = camera_drone_tr
        extrinsic_matrix_drone_camera = rotation.get_extrinsic_matrix(rotation_matrix_drone_camera, translation_vector=dc_tr)

        extrinsic_matrix = extrinsic_matrix_world_drone@extrinsic_matrix_drone_camera
                    ########### ------------------ ROTATIONS ------------------ ###########

        # Twist velocity from camera frame to world frame
        rot = np.array(extrinsic_matrix)[:3,:3] # from World to Camera
        t = np.array(extrinsic_matrix)[:3,-1] # translation vector world camera
        twist_drone_camera = geometry.velocity_twist_matrix(rotation_matrix_drone_camera.T, dc_tr)
        v_drone = twist_drone_camera.T@v_camera
        v_x, v_y, v_z, w_x, w_y, w_z = v_drone
        
        ########### ------------------ VISUAL SERVOING ------------------ ###########
        
        ## Fill in Setpoints
        setpoint = cffirmware.setpoint_t()
        
        setpoint.mode.yaw = cffirmware.modeVelocity
        # Not working 
        magic_constant = 0.0626 # Value found using proportions since yaw command is lower than the one requested.
        setpoint.attitudeRate.yaw = degrees(w_z/magic_constant)
        
        
        setpoint.mode.x = cffirmware.modeVelocity
        setpoint.mode.y = cffirmware.modeVelocity
        setpoint.mode.z = cffirmware.modeVelocity
        
        setpoint.velocity.x = v_x
        setpoint.velocity.y = v_y
        setpoint.velocity.z = v_z
        
        setpoint.velocity_body = True # False for velocities in world frame. True for velocities boxy-fixed
    
        ## Firmware PID bindings
        control = cffirmware.control_t()
        tick = 100 #this value makes sure that the position controller and attitude controller are always always initiated
        cffirmware.controllerPid(control, setpoint,sensors,state,tick)

        ## 
        cmd_roll = radians(control.roll)
        cmd_pitch = radians(control.pitch)
        cmd_yaw = -radians(control.yaw)
        cmd_thrust = control.thrust

        ## Motor mixing
        motorPower_m1 =  cmd_thrust - cmd_roll + cmd_pitch + cmd_yaw
        motorPower_m2 =  cmd_thrust - cmd_roll - cmd_pitch - cmd_yaw
        motorPower_m3 =  cmd_thrust + cmd_roll - cmd_pitch + cmd_yaw
        motorPower_m4 =  cmd_thrust + cmd_roll + cmd_pitch - cmd_yaw

        scaling = 1000 ##Todo, remove necessity of this scaling (SI units in firmware)
        scaling = 600 # DIFFERENT SCALING FACTOR
        m1_motor.setVelocity(-motorPower_m1/scaling)
        m2_motor.setVelocity(motorPower_m2/scaling)
        m3_motor.setVelocity(-motorPower_m3/scaling)
        m4_motor.setVelocity(motorPower_m4/scaling)
        
        past_time = robot.getTime()
        pastXGlobal = xGlobal
        pastYGlobal = yGlobal
        pastZGlobal = zGlobal
        
        print(f'vx:{v_x:.4f}\tvy:{v_y:.4f}\tvz:{v_z:.4f}')
        print(f'wz:{w_z:.4f}, gyro_w_z:{yaw_rate:.4f}')
        
    else:
        # keyboard input
        forwardDesired = 0
        sidewaysDesired = 0
        yawDesired = 0
        goUpDesired = 0

        key = keyboard.getKey()
        while key>0:
            if key == Keyboard.UP:
                forwardDesired = 0.5
            elif key == Keyboard.DOWN:
                forwardDesired = -0.5
            elif key == Keyboard.RIGHT:
                sidewaysDesired = -0.5
            elif key == Keyboard.LEFT:
                sidewaysDesired = 0.5
            elif key == ord('Q'):
                yawDesired = 8
            elif key == ord('E'):
                yawDesired = -8
            elif key == ord('W'):
                goUpDesired = 0.2
            elif key == ord('S'):
                goUpDesired = -0.2

            key = keyboard.getKey()

        ## Example how to get sensor data
        # range_front_value = range_front.getValue();
        # cameraData = camera.getImage()

        ## Fill in Setpoints
        setpoint = cffirmware.setpoint_t()
        setpoint.mode.z = cffirmware.modeAbs
        setpoint.position.z = 1
        setpoint.mode.yaw = cffirmware.modeAbs
        setpoint.attitude.yaw = degrees(np.pi/2) # TODO: change in starting yaw
        
        #setpoint.mode.yaw = cffirmware.modeVelocity
        #setpoint.attitudeRate.yaw = degrees(yawDesired)
        # setpoint.mode.x = cffirmware.modeVelocity
        # setpoint.mode.y = cffirmware.modeVelocity
        # setpoint.mode.z = cffirmware.modeVelocity
        # setpoint.velocity.x = forwardDesired
        # setpoint.velocity.y = sidewaysDesired
        # setpoint.velocity.z = goUpDesired
        setpoint.velocity_body = True

        ## Firmware PID bindings
        control = cffirmware.control_t()
        tick = 100 #this value makes sure that the position controller and attitude controller are always always initiated
        cffirmware.controllerPid(control, setpoint,sensors,state,tick)

        ## 
        cmd_roll = radians(control.roll)
        cmd_pitch = radians(control.pitch)
        cmd_yaw = -radians(control.yaw)
        cmd_thrust = control.thrust
        
        ## Motor mixing
        motorPower_m1 =  cmd_thrust - cmd_roll + cmd_pitch + cmd_yaw
        motorPower_m2 =  cmd_thrust - cmd_roll - cmd_pitch - cmd_yaw
        motorPower_m3 =  cmd_thrust + cmd_roll - cmd_pitch + cmd_yaw
        motorPower_m4 =  cmd_thrust + cmd_roll + cmd_pitch - cmd_yaw

        scaling = 1000 ##Todo, remove necessity of this scaling (SI units in firmware)
        
        m1_motor.setVelocity(-motorPower_m1/scaling)
        m2_motor.setVelocity(motorPower_m2/scaling)
        m3_motor.setVelocity(-motorPower_m3/scaling)
        m4_motor.setVelocity(motorPower_m4/scaling)
        
        past_time = robot.getTime()
        pastXGlobal = xGlobal
        pastYGlobal = yGlobal
        pastZGlobal = zGlobal
        
        # if state.position.z >= 1:
        #    start_vs = True
        #    print("SET TO TRUE")
        
        start_vs = True
        print("SET TO TRUE")
        
        print(f'vx:{forwardDesired:.4f}\tvy:{sidewaysDesired:.4f}\tvz:{goUpDesired:.4f}')
        