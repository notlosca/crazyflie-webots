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
from pid_controller import pid_velocity_fixed_height_controller

FLYING_ATTITUDE = 1

if __name__ == '__main__':

    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())

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

    ## Initialize variables

    past_x_global, past_y_global = (None,None)
    past_z_global = None
    past_time = robot.getTime()

    # Crazyflie velocity PID controller
    PID_CF = pid_velocity_fixed_height_controller()
    PID_update_last_time = robot.getTime()
    sensor_read_last_time = robot.getTime()

    # Velocity PID control (converted from Crazyflie c code)
    # Original ones
    gains = {"kp_att_y": 1, "kd_att_y": 0.5, "kp_att_rp": 0.5, "kd_att_rp": 0.1,
             "kp_vel_xy": 2, "kd_vel_xy": 0.5, "kp_z": 10, "ki_z": 5, "kd_z": 5}
    # Mine
    # gains = {"kp_att_y": 1, "kd_att_y": 0.5, "kp_att_rp": 0.5, "kd_att_rp": 0.1,
    #          "kp_vel_xy": 2, "kd_vel_xy": 0.5, "kp_z": 8, "ki_z": 5, "kd_z": 5}

    height_desired = FLYING_ATTITUDE
    height_desired = 0
    print("\n");

    print("====== Controls =======\n\n");

    print(" The Crazyflie can be controlled from your keyboard!\n");
    print(" All controllable movement is in body coordinates\n");
    print("- Use the up, back, right and left button to move in the horizontal plane\n");
    print("- Use Q and E to rotate around yaw ");
    print("- Use W and S to go up and down\n ")
    
    it_idx = 0 # Iteration index. Used to set the initial status at the first iteration.
        
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

    # Main loop:
    while robot.step(timestep) != -1 and err >= thresh:

        dt = robot.getTime() - past_time
        actual_state = {}

        ## Get sensor data
        roll, pitch, yaw = imu.getRollPitchYaw()
        roll_rate, pitch_rate, yaw_rate = gyro.getValues()
        altitude = gps.getValues()[2] # z_global
        x_global, y_global, z_global = gps.getValues()
        

        if it_idx == 0:
            starting_roll, starting_pitch, starting_yaw = roll, pitch, yaw
            # Initialization
            v_x_global = (x_global - x_global)/dt
            v_y_global = (y_global - y_global)/dt
            v_z_global = (z_global - z_global)/dt
            it_idx += 1
        
        else:
            v_x_global = (x_global - past_x_global)/dt
            v_y_global = (y_global - past_y_global)/dt
            v_z_global = (z_global - past_z_global)/dt

        ## Get body fixed velocities
        cosyaw = cos(yaw)
        sinyaw = sin(yaw)
        v_x_body = v_x_global * cosyaw + v_y_global * sinyaw
        v_y_body = - v_x_global * sinyaw + v_y_global * cosyaw

        ## Initialize values
        desired_state = [0, 0, 0, 0]
        forward_desired = 0
        sideways_desired = 0
        yaw_desired = 0
        height_diff_desired = 0

        # CAMERA IMAGES
        w, h = camera.getWidth(), camera.getHeight()
        cameraData = camera.getImage()  # Note: uint8 string
        image = np.frombuffer(cameraData, np.uint8).reshape(h, w, 4) # BGR, alpha (transparency)
        
        
        ########### ------------------ ROTATIONS ------------------ ###########
        rotation_matrix_world_drone = rotation.rotation_matrix(roll, pitch, yaw)
        wd_tr = np.array([x_global, y_global, z_global])
        extrinsic_matrix_world_drone = rotation.get_extrinsic_matrix(rotation_matrix_world_drone, translation_vector=wd_tr)
        
        rotation_matrix_drone_camera = rotation.rotation_matrix(roll=-np.pi/2, pitch=0, yaw=-np.pi/2)
        dc_tr = camera_drone_tr
        extrinsic_matrix_drone_camera = rotation.get_extrinsic_matrix(rotation_matrix_drone_camera, translation_vector=dc_tr)

        extrinsic_matrix = extrinsic_matrix_world_drone@extrinsic_matrix_drone_camera
        ########### ------------------ ROTATIONS ------------------ ###########
        
        # print(roll, pitch, yaw)
        
        T_C = SE3(wd_tr)*SE3.RPY(roll,pitch,yaw)*SE3(dc_tr)*SE3.RPY(-np.pi/2, 0, -np.pi/2)
        # print(T_C)
        # print(wd_tr)
        p_detected = cam.project_point(P, pose=SE3(T_C, check=False)) 
        # print(p_detected)
        ########### ------------------ VISUAL SERVOING ------------------ ###########
        # image-plane error
        try:
            e = pd - p_detected
            err = np.linalg.norm(e)
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
        # print('v_x, v_y, v_z, w_x, w_y, w_z', v_x, v_y, v_z, w_x, w_y, w_z)
        forward_desired = v_x
        sideways_desired = v_y
        yaw_desired = w_z
        height_diff_desired = v_z

        ########### ------------------ VISUAL SERVOING ------------------ ###########

        # key = keyboard.getKey()
        # while key>0:
        #     if key == Keyboard.UP:
        #         forward_desired += 0.5
        #     elif key == Keyboard.DOWN:
        #         forward_desired -= 0.5
        #     elif key == Keyboard.RIGHT:
        #         sideways_desired -= 0.5
        #     elif key == Keyboard.LEFT:
        #         sideways_desired += 0.5
        #     elif key == ord('Q'):
        #         yaw_desired =  + 1
        #     elif key == ord('E'):
        #         yaw_desired = - 1
        #     elif key == ord('W'):
        #         height_diff_desired = 0.1
        #     elif key == ord('S'):
        #         height_diff_desired = - 0.1
        # 
        #     key = keyboard.getKey()


        height_desired += height_diff_desired * dt

        ## Example how to get sensor data
        ## range_front_value = range_front.getValue();
        ## cameraData = camera.getImage()


        ## PID velocity controller with fixed height
        motor_power = PID_CF.pid(dt, forward_desired, sideways_desired,
                                yaw_desired, height_desired,
                                roll, pitch, yaw_rate,
                                altitude, v_x_body, v_y_body, gains)

        m1_motor.setVelocity(-motor_power[0])
        m2_motor.setVelocity(motor_power[1])
        m3_motor.setVelocity(-motor_power[2])
        m4_motor.setVelocity(motor_power[3])

        past_time = robot.getTime()
        past_x_global = x_global
        past_y_global = y_global
        past_z_global = z_global
        
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

        ########### ------------------ PRINT -------------------- ########### 
        print("########### ------------------ GPS [m] -------------------- ###########")
        print(f"X: {x_global:.4f}\tY: {y_global:.4f}\tZ: {z_global:.4f}")
        print("########### ------------------ GPS [m] -------------------- ###########")
        print("\n ")
        print("########### ------------------ VELOCITIES [m/s] -------------------- ###########")
        print(f"X: {v_x_global:.4f}\tY: {v_y_global:.4f}\tZ: {v_z_global:.4f}")
        print("########### ------------------ VELOCITIES [m/s] -------------------- ###########")
        print("\n ")
        # print("########### ------------------ BODY VELOCITIES [m/s] -------------------- ###########")
        # print(f"X: {v_x:.4f}\tY: {v_y:.4f}\tZ: NO")
        # print("########### ------------------ BODY VELOCITIES [m/s] -------------------- ###########")
        # print("\n ")
        print("########### ------------------ IMU [rad] -------------------- ###########")
        print(f"R: {roll:.4f}\tP: {pitch:.4f}\tY: {yaw:.4f}")
        print("########### ------------------ IMU [rad] -------------------- ###########")
        print("\n ")
        print(f"########### ------------------ ATTITUDE RATES [rad/s] -------------------- ###########")
        print(f"R: {roll_rate:.4f}\tP: {pitch_rate:.4f}\tY: {yaw_rate:.4f}")
        print(f"########### ------------------ ATTITUDE RATES [rad/s] -------------------- ###########")
        print("\n ")
        print(f"########### ------------------ VISUAL SERVOING ERROR -------------------- ###########")
        print(f"Error: {err}")
        print(f"########### ------------------ VISUAL SERVOING ERROR -------------------- ###########")
        print("\n ")
        ########### ------------------ PRINT -------------------- ########### 

        cnt += 1