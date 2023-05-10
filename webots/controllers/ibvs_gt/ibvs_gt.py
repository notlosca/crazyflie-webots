# Take off and IBVS tasks


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

import sys
sys.path.append('../../../controllers/')

# Import the path for the corner detection module
sys.path.append('../../../../scarciglia-nanodrone-gate-detection/')

from src import corner, rotation, geometry

# Change this path to your crazyflie-firmware folder
sys.path.append('../../../../crazyflie-firmware/build')

import cffirmware

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

# print(m1_motor.getTargetPosition())
# print(m2_motor.getVelocity())
# print(m3_motor.getVelocity())
# print(m4_motor.getVelocity())

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
gate_rot = gate_node.getField('rotation').getSFRotation()

# Gate pose. This is used to transform the gate points.
T_G = SE3(translation_gate)*SE3(transforms3d.axangles.axangle2aff(gate_rot[:-1], gate_rot[-1]))

br = gate_node.getField('children').getMFNode(0).getField('translation').getSFVec3f()
bl = gate_node.getField('children').getMFNode(1).getField('translation').getSFVec3f()
tl = gate_node.getField('children').getMFNode(2).getField('translation').getSFVec3f()
tr = gate_node.getField('children').getMFNode(3).getField('translation').getSFVec3f()

# Rotate gate points. Transform in homogenous coordinates, perform the matmul, and then pick the first three entries.
tl = (np.array(T_G)@np.append(tl, 1))[:-1]
bl = (np.array(T_G)@np.append(bl, 1))[:-1]
br = (np.array(T_G)@np.append(br, 1))[:-1]
tr = (np.array(T_G)@np.append(tr, 1))[:-1]

P = np.array([
    tl, bl, br, tr
]).T
print('Gate data (br, bl, tl, tr, gate_center, translation_gate):', br, bl, tl, tr, gate_center, translation_gate)

## Get keyboard
keyboard = Keyboard()
keyboard.enable(timestep)

# Main loop:

## Initialize variables
# actualState = ActualState_t()
# desiredState = DesiredState_t()
pastXGlobal = 0
pastYGlobal = 0
pastZGlobal = 0
past_time = robot.getTime()

cffirmware.controllerPidInit()

# OpenCV show images
cv2.startWindowThread()
cv2.namedWindow("preview")
colors = ['r', 'b', 'g', 'y']

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

print(f'[POINTS DESIRED]\n{pd}')

lmda = 0.08

thresh = 5e-1
err = np.inf

########### ------------------ VISUAL SERVOING ------------------ ###########

starting_roll = 0
starting_pitch = 0
starting_yaw = 0
cnt = 0
# starting_yaw = np.pi/2

visual_servoing_task = False
takeoff_height = 1.0
prev_step = False
hovering_steps = 0 # Counter used to check how many times we are in the desired state.

while robot.step(timestep) != -1 and err > thresh: # and frame_n < n_samples: # with this condition the controller will exit
        
    dt = robot.getTime() - past_time
    
    ## Get measurements
    roll, pitch, yaw = imu.getRollPitchYaw() # rad
    if cnt == 0:
        starting_roll, starting_pitch, starting_yaw = roll, pitch, yaw
    roll_rate, pitch_rate, yaw_rate = gyro.getValues() # rad/s
    xGlobal, yGlobal, zGlobal = gps.getValues()
    vxGlobal = (xGlobal - pastXGlobal)/dt
    vyGlobal = (yGlobal - pastYGlobal)/dt
    vzGlobal = (zGlobal - pastZGlobal)/dt
    
    print(f"[GPS]\tX:{xGlobal:.4f}\tY:{yGlobal:.4f}\tZ:{zGlobal:.4f}")
    
    print(f"[VELOCITIES]\tvX:{vxGlobal:.4f}\tvY:{vyGlobal:.4f}\tvZ:{vzGlobal:.4f}")
    
    print(f"[IMU]\tRoll:{roll:.4f}\tPitch:{pitch:.4f}\tYaw:{yaw:.4f}")
    
    ## Put measurement in state estimate
    # TODO replace these with a EKF python binding
    state = cffirmware.state_t()
    state.attitude.roll = degrees(roll) # - starting_roll)
    state.attitude.pitch = -degrees(pitch) # - starting_pitch) 
    state.attitude.yaw = degrees(yaw) # - starting_yaw)
    state.position.x = xGlobal
    state.position.y = yGlobal
    state.position.z = zGlobal
    state.velocity.x = vxGlobal
    state.velocity.y = vyGlobal
    state.velocity.z = vzGlobal
    
    # Put gyro in sensor data
    sensors = cffirmware.sensorData_t()
    sensors.gyro.x = degrees(roll_rate) # from rad/s to deg/s
    sensors.gyro.y = degrees(pitch_rate)
    sensors.gyro.z = degrees(yaw_rate)
    
    # # keyboard input
    # forwardDesired = 0
    # sidewaysDesired = 0
    # yawDesired = yaw
    # goUpDesired = 0

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
    
    # print(roll, pitch, yaw)
    
    T_C = SE3(wd_tr)*SE3.RPY(roll,pitch,yaw)*SE3(dc_tr)*SE3.RPY(-np.pi/2, 0, -np.pi/2)
    # print(T_C)
    # print(wd_tr)
    p_detected = cam.project_point(P, pose=SE3(T_C, check=False)) 
    # print(p_detected)
    ########### ------------------ VISUAL SERVOING ------------------ ###########
    # image-plane error
    try:
        e = pd - p_detected
    except:
        continue
    # Z_l = []
    # for i in range(P_hom.shape[-1]):
    #     Z_l.append((np.linalg.inv(T_C) @ P_hom[:,i])[-2])
    # 
    # #print(Z_l)
    # 
    # Z = np.array(Z_l)
    
    # stacked image Jacobian
    J = cam.visjac_p(p_detected, Z)
    #J = cam.visjac_p(p, 0.35)
    v_camera = lmda * np.linalg.pinv(J) @ e.T.flatten()

    # Twist velocity from camera frame to drone frame
    # twist_world_camera = geometry.velocity_twist_matrix(extrinsic_matrix[:3,:3].T, extrinsic_matrix[:3,-1])
    # v_world = twist_world_camera.T@v_camera
    # v_x, v_y, v_z, w_x, w_y, w_z = v_world
    twist_drone_camera = geometry.velocity_twist_matrix(rotation_matrix_drone_camera, dc_tr)
    v_drone = twist_drone_camera@v_camera
    v_x, v_y, v_z, w_x, w_y, w_z = v_drone
    
    ########### ------------------ VISUAL SERVOING ------------------ ###########
    
    if visual_servoing_task:
        
        print("[IBVS] err:", np.linalg.norm(e))
        
        ## Fill in Setpoints
        setpoint = cffirmware.setpoint_t()
        
        setpoint.mode.yaw = cffirmware.modeVelocity
        setpoint.attitudeRate.yaw = degrees(w_z)*100

                
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
        # scaling = 600 # DIFFERENT SCALING FACTOR
        m1_motor.setVelocity(-motorPower_m1/scaling)
        m2_motor.setVelocity(motorPower_m2/scaling)
        m3_motor.setVelocity(-motorPower_m3/scaling)
        m4_motor.setVelocity(motorPower_m4/scaling)
    
    else:
        isclose = np.isclose(zGlobal, takeoff_height, rtol=1e-1)
        if isclose and prev_step:
            print(True)
            print("prev_step", prev_step)
            if prev_step:
                hovering_steps += 1
                prev_step = isclose
                if hovering_steps >= 100:
                    visual_servoing_task = True
        else:
            hovering_steps = 0
            prev_step = isclose
        
        print(hovering_steps)
        
        ## Fill in Setpoints
        setpoint = cffirmware.setpoint_t()
        
        setpoint.mode.z = cffirmware.modeAbs
        setpoint.position.z = takeoff_height # 1.0

        setpoint.mode.x = cffirmware.modeVelocity
        setpoint.mode.y = cffirmware.modeVelocity
        setpoint.velocity.x = 0.0
        setpoint.velocity.y = 0.0
        
        setpoint.mode.yaw = cffirmware.modeVelocity
        setpoint.attitudeRate.yaw = 0

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
        # print('cmd_roll', cmd_roll)
        # print('cmd_pitch', cmd_pitch)
        # print('cmd_yaw', cmd_yaw)
        # print('cmd_thrust', cmd_thrust)

        ## Motor mixing
        motorPower_m1 =  cmd_thrust - cmd_roll + cmd_pitch + cmd_yaw
        motorPower_m2 =  cmd_thrust - cmd_roll - cmd_pitch - cmd_yaw
        motorPower_m3 =  cmd_thrust + cmd_roll - cmd_pitch + cmd_yaw
        motorPower_m4 =  cmd_thrust + cmd_roll + cmd_pitch - cmd_yaw
        
        # print('motorPower_m1', motorPower_m1)
        # print('motorPower_m2', motorPower_m2)
        # print('motorPower_m3', motorPower_m3)
        # print('motorPower_m4', motorPower_m4)
        
        scaling = 1000 ##Todo, remove necessity of this scaling (SI units in firmware)
        # scaling = 600 # DIFFERENT SCALING FACTOR
        m1_motor.setVelocity(-motorPower_m1/scaling)
        m2_motor.setVelocity(motorPower_m2/scaling)
        m3_motor.setVelocity(-motorPower_m3/scaling)
        m4_motor.setVelocity(motorPower_m4/scaling)
        
        # print(m1_motor.getVelocity())
        # print(m2_motor.getVelocity())
        # print(m3_motor.getVelocity())
        # print(m4_motor.getVelocity())
    
    past_time = robot.getTime()
    pastXGlobal = xGlobal
    pastYGlobal = yGlobal
    pastZGlobal = zGlobal
    
    # print(f'vx:{v_x:.4f}\tvy:{v_y:.4f}\tvz:{v_z:.4f}')
    # print(f'wx:{w_x:.4f}\twy:{w_y:.4f}\twz:{w_z:.4f}')
    
    # print('starting_yaw', starting_yaw)
    
    # Show image
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

    cv2.imshow("preview", cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    cv2.waitKey(timestep)

    cnt += 1