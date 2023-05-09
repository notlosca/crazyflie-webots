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


from controller import Robot
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
random.seed()

from math import cos, sin

import sys
sys.path.append('../../../controllers/')
##### My ADDITION #####
# print(sys.path)
# print(os.getcwd())
# print(os.listdir('../../../controllers/'))
##### My ADDITION #####
from  pid_controller import init_pid_attitude_fixed_height_controller, pid_velocity_fixed_height_controller
from pid_controller import MotorPower_t, ActualState_t, GainsPID_t, DesiredState_t
robot = Robot()

# Set to True if you want to collect data
collect_data = False

dataset_path = '../../datasets/EXP-1-NOPHYSICS-SPHERE'
imgs_folder = f'{dataset_path}/imgs/'
try:
    os.makedirs(imgs_folder)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise  # This was not a "directory exist" error..
    else:
        if collect_data:
            print(f"Overwriting folder {imgs_folder}")

timestep = int(robot.getBasicTimeStep()) # original
#timestep = 100 # in ms. This will make the drone not to fly

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
Keyboard().enable(timestep)
gyro = robot.getDevice("gyro")
gyro.enable(timestep)
camera = robot.getDevice("camera")
camera.enable(timestep)

####
gps_camera = robot.getDevice("gps_camera")
gps_camera.enable(timestep)
imu_camera = robot.getDevice("imu_camera")
imu_camera.enable(timestep)
####

range_front = robot.getDevice("range_front")
range_front.enable(timestep)
range_left = robot.getDevice("range_left")
range_left.enable(timestep)
range_back = robot.getDevice("range_back")
range_back.enable(timestep)
range_right = robot.getDevice("range_right")
range_right.enable(timestep)
    
## Initialize variables
actualState = ActualState_t()
desiredState = DesiredState_t()
pastXGlobal = 0
pastYGlobal = 0
past_time = robot.getTime()

## Initialize PID gains.
gainsPID = GainsPID_t()
gainsPID.kp_att_y = 1
gainsPID.kd_att_y = 0.5
gainsPID.kp_att_rp =0.5
gainsPID.kd_att_rp = 0.1
gainsPID.kp_vel_xy = 2
gainsPID.kd_vel_xy = 0.5
gainsPID.kp_z = 10   # [PRODUCT] 10 original. Same as integral.
gainsPID.ki_z = 50   # [INTEGRAL] 50 original. Under 50 it doesn't take off. With 60 it goes forward (y direction)
gainsPID.kd_z = 7.5  # [DERIVATIVE] 5 original. The lower the value, the greater it overshoots the target height.
init_pid_attitude_fixed_height_controller()

## Speeds
forward_speed = 0.2
yaw_rate = 0.5

## Avoidance state
avoid_yawDesired = 0
avoid_yawTime = 0

## Initialize struct for motor power
motorPower = MotorPower_t()

print('Take off!')

# Main loop:
frame_n = 0
x_lim = (-3, +3)
y_lim = (-5, +0)
z_lim = (0, +2)


###### GRID to set number of samples #######
# create a grid around the gate
x_grid = np.linspace(x_lim[0], x_lim[1], num=2*(x_lim[1] - x_lim[0] + 1), endpoint=True)
y_grid = np.linspace(y_lim[0], y_lim[1], num=2*(y_lim[1] - y_lim[0] + 1), endpoint=True)
z_grid = np.linspace(z_lim[0], z_lim[1], num=2*(z_lim[1] - z_lim[0] + 1), endpoint=True)
grid_coord = list(itertools.product(x_grid,y_grid,z_grid))

n_samples = len(grid_coord)
###### GRID to set number of samples #######

###### SPHERE to set number of samples #######
theta_lim = (0,np.pi)
phi_lim = (0,2*np.pi)

r = 2
thetas = list(np.linspace(theta_lim[0], theta_lim[1], num=50))
phis = list(np.linspace(phi_lim[0], phi_lim[1], num=50))

array = np.array([0,0,0])

for pair in list(itertools.product(thetas, phis)):
    t = pair[0]
    p = pair[1]
    x = r*np.sin(t)*np.cos(p)
    y = r*np.sin(t)*np.sin(p)
    z = r*np.cos(t)
    array = np.vstack((array, np.array([x,y,z])))

array = array[1:] # remove first entry
mask = array[:, -1] >= 0
clean_array = array[mask]

#n_samples = len(clean_array)
###### SPHERE to set number of samples #######

samples = []
saved_data = False
# header = ['drone_imu_rpy', 'drone_gyro_wxwywz', 'drone_gps_xyz', 'drone_camera_gps_xyz']
while robot.step(timestep) != -1: # and frame_n < n_samples: # with this condition the controller will exit

    dt = robot.getTime() - past_time;
    
    ## Get measurements
    actualState.roll = imu.getRollPitchYaw()[0]
    actualState.pitch = imu.getRollPitchYaw()[1]
    actualState.yaw_rate = gyro.getValues()[2];
    actualState.altitude = gps.getValues()[2];
    xGlobal = gps.getValues()[0]
    vxGlobal = (xGlobal - pastXGlobal)/dt
    yGlobal = gps.getValues()[1]
    vyGlobal = (yGlobal - pastYGlobal)/dt

    ### MY ADDITION ###
    zGlobal = gps.getValues()[2] # same as actualState.altitude
    # print(f"[GPS]\tX:{xGlobal:.4f}\tY:{yGlobal:.4f}\tZ:{zGlobal:.4f}")

    roll, pitch, yaw = imu.getRollPitchYaw()
    # print(f"[IMU]\tRoll:{roll:.4f}\tPitch:{pitch:.4f}\tYaw:{yaw:.4f}")
    
    # gps_camera_pos = gps_camera.getValues()
    # imu_camera_rpy = imu_camera.getRollPitchYaw()
    # print(f"[GPS - CAMERA]\tX:{gps_camera_pos[0]:.4f}\tY:{gps_camera_pos[1]:.4f}\tZ:{gps_camera_pos[2]:.4f}")
    
    if collect_data and frame_n < n_samples:
        sample = {}
        # SAVE MEASUREMENTS
        # sample['frame'] = f"img_{frame_n}.png" # not necessary anymore. the json is now saved in img:{data} format
        sample['drone_imu_rpy'] = imu.getRollPitchYaw()
        sample['drone_gyro_wxwywz'] = gyro.getValues() # angular velocity around x (= [0]), y (= [1]), z (= [2])
        sample['drone_gps_xyz'] = gps.getValues()
        sample['drone_camera_gps_xyz'] = gps_camera.getValues()
        sample['drone_camera_imu_rpy'] = imu_camera.getRollPitchYaw()
        
        samples.append({f"img_{frame_n}.png": sample})
        
        # CAMERA IMAGES
        w, h = camera.getWidth(), camera.getHeight()
        cameraData = camera.getImage()  # Note: uint8 string
    
        image = np.frombuffer(cameraData, np.uint8).reshape(h, w, 4)
        #imgs_folder = "/home/lorenzo/crazyflie-simulation/my_project/datasets/EXP-0"
        path = f"{imgs_folder}/img_{frame_n}.png"
        cv2.imwrite(path, image)
        print("Frame n:", frame_n)
        frame_n += 1
    
    #print(f"[CAMERA]\t{image}")
    ### MY ADDITION ###

    ## Get body fixed velocities
    actualYaw = imu.getRollPitchYaw()[2];
    cosyaw = cos(actualYaw)
    sinyaw = sin(actualYaw)
    actualState.vx = vxGlobal * cosyaw + vyGlobal * sinyaw
    actualState.vy = - vxGlobal * sinyaw + vyGlobal * cosyaw

    ## Initialize setpoints
    desiredState.roll = 0
    desiredState.pitch = 0
    desiredState.vx = 0
    desiredState.vy = 0
    desiredState.yaw_rate = 0
    desiredState.altitude = 1.5

    forwardDesired = 0
    sidewaysDesired = 0
    yawDesired = 0

    ## Get camera image
    w, h = camera.getWidth(), camera.getHeight()
    cameraData = camera.getImage()  # Note: uint8 string
    image = np.frombuffer(cameraData, np.uint8).reshape(h, w, 4)
    
    # Show image
    # cv2.imshow('Drone camera', image)
    # cv2.waitKey(1)

    ## Detect empty floor (green) in front of the drone
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    xmin, xmax = int(0.20 * w), int(0.80 * w)
    ymin, ymax = int(0.80 * h), int(1.00 * h)
    max_pix = (xmax - xmin) * (ymax - ymin)
    hmin, hmax = 30, 46
    roi_hue = image[ymin:ymax, xmin:xmax, 0]
    roi_sat = image[ymin:ymax, xmin:xmax, 1]
    pix_count = np.count_nonzero((roi_hue >= hmin) & (roi_hue <= hmax) & (roi_sat > 64))
    green_pct = pix_count / max_pix

    # ## Avoidance state machine 
    # if avoid_yawTime > 0:
    #     # Turning
    #     avoid_yawTime -= dt
    #     yawDesired += avoid_yawDesired
    # else:
    #     # Not turning
    #     if green_pct > 0.20:
    #         # No obstacle: fly forwards
    #         forwardDesired += forward_speed
    #         turn_rate = 0
    #     else:
    #         # Obstacle in front: start turn
    #         sign = 1 if random.random() > 0.5 else -1
    #         avoid_yawDesired = sign * yaw_rate
    #         avoid_yawTime = random.random() * 5.0

    # Manual override
    key = Keyboard().getKey()
    while key>0:
        if key == Keyboard.UP:
            forwardDesired = forward_speed
        elif key == Keyboard.DOWN:
            forwardDesired = -forward_speed
        elif key == Keyboard.RIGHT:
            sidewaysDesired  = -forward_speed
        elif key == Keyboard.LEFT:
            sidewaysDesired = forward_speed
        elif key == ord('Q'):
            yawDesired =  + yaw_rate
        elif key == ord('E'):
            yawDesired = - yaw_rate

        key = Keyboard().getKey()

    desiredState.yaw_rate = yawDesired;

    ## PID velocity controller with fixed height
    desiredState.vy = sidewaysDesired;
    desiredState.vx = forwardDesired;
    pid_velocity_fixed_height_controller(actualState, desiredState, gainsPID, dt, motorPower);

    m1_motor.setVelocity(-motorPower.m1)
    m2_motor.setVelocity(motorPower.m2)
    m3_motor.setVelocity(-motorPower.m3)
    m4_motor.setVelocity(motorPower.m4)
    
    past_time = robot.getTime()
    pastXGlobal = xGlobal
    pastYGlobal = yGlobal

    if collect_data and frame_n == n_samples and not saved_data:

        print("!STOP!")

        with open(f"{dataset_path}/drone.json", "w") as f:
            json.dump(samples , f)
        saved_data = True



