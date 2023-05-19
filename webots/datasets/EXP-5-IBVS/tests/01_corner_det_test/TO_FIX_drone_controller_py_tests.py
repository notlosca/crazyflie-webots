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

########### ------------------ SAVING THINGS -------------------- ###########
    
# Set to True if you want to collect data
collect_data = False

if collect_data:
        
    parent_folder = '../../datasets/EXP-5-IBVS'
    folder = parent_folder +'/tests/'+ '01_corner_det_test'

    imgs_folder = f'{folder}/imgs/'
    imgs_ibvs_folder = f'{folder}/imgs_ibvs/'

    if not os.path.isdir(folder):
        os.makedirs(folder)

    if not os.path.isdir(imgs_folder):
        os.makedirs(imgs_folder)

    if not os.path.isdir(imgs_ibvs_folder):
        os.makedirs(imgs_ibvs_folder)

########### ------------------ SAVING THINGS -------------------- ###########

if __name__ == '__main__':

    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    
    sampling_frequency = timestep
    
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

    print("\n");

    print("====== Controls =======\n\n");

    print(" The Crazyflie can be controlled from your keyboard!\n");
    print(" All controllable movement is in body coordinates\n");
    print("- Use the up, back, right and left button to move in the horizontal plane\n");
    print("- Use Q and E to rotate around yaw ");
    print("- Use W and S to go up and down\n ")
       
    ########### ------------------ HOVERING ------------------ ###########

    # Hovering
    prev_step = False
    hovering_time = 3 # seconds
    hovering_steps = hovering_time*sampling_frequency
    
    ########### ------------------ HOVERING ------------------ ###########
    
    ########### ------------------ TASKS ------------------ ###########
    
    # Desired states
    tasks = {}

    tasks['order'] = ['take_off', 'visual_servoing', 'cross_the_gate', 'land']

    take_off = True
    take_off_info = {'setpoints': {'velocity.x':0.0, 'velocity.y':0.0, 'position.z':1, 'attitudeRate.yaw':0.0}}
    tasks['take_off'] = take_off_info

    visual_servoing = False
    old_p_detected = None
    vs_init = False # To initialize the old_p_detected
    tasks['visual_servoing'] = {'visual_servoing':True}
    
    cross_the_gate = False
    num_seconds = 10
    cross_the_gate_info = {'setpoints': {'velocity.x':0.1, 'velocity.y':0.0, 'velocity.z':0.0, 'attitudeRate.yaw':0.0}, 'num_steps':num_seconds*sampling_frequency}
    cross_the_gate_steps = 0
    tasks['cross_the_gate'] = cross_the_gate_info

    landing = False
    landing_info = {'setpoints': {'velocity.x':0.0, 'velocity.y':0.0, 'velocity.z':-0.1, 'attitudeRate.yaw':0.0}}
    landing_steps = 0
    tasks['land'] = landing_info

    dataset = {'info':None, 'sampling_frequency':sampling_frequency, 'corner_detection': False}

    ########### ------------------ TASKS ------------------ ###########

    # OpenCV show images
    cv2.startWindowThread()
    cv2.namedWindow("Drone Camera")
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
    thresh = 5
    err = np.inf

    ########### ------------------ VISUAL SERVOING ------------------ ###########

    ########### ------------------ SAVING THINGS ------------------ ###########
    
    dataset['drone'] = {'starting_position': crazyflie_node.getField('translation').getSFVec3f(),
                        'starting_rotation': crazyflie_node.getField('rotation').getSFRotation(),
                        'camera_drone_tr': camera_node.getField('translation').getSFVec3f()}
    dataset['gate'] = {'corners':{'tl':tl, 'bl':bl, 'br':br, 'tr':tr},
                    'position': gate_node.getField('translation').getSFVec3f(),
                        'rotation': gate_node.getField('rotation').getSFRotation()}
    dataset['camera'] = {'f':f, 'pixel_size':pixel_size, 'img_size':img_size }
    dataset['ibvs'] = {'lambda': lmda, 'threshold': thresh, 'Z':{'estimated':False, 'value':Z}}
    
    ########### ------------------ SAVING THINGS ------------------ ###########

    height_desired = take_off_info['setpoints']['position.z']

    ## Initialize values
    desired_state = [0, 0, 0, 0] # Not used
    forward_desired = 0
    sideways_desired = 0
    yaw_desired = 0
    height_diff_desired = 0
    starting_altitude = None
    
    it_idx = 0 # Iteration index
    
    print("Take off!")
    
    # Main loop:
    while robot.step(timestep) != -1:

        data = {}
        
        dt = robot.getTime() - past_time
        # actual_state = {} # Not used

        ## Get sensor data
        roll, pitch, yaw = imu.getRollPitchYaw()
        roll_rate, pitch_rate, yaw_rate = gyro.getValues()
        altitude = gps.getValues()[2]
        x_global, y_global, z_global = gps.getValues()
        # v_x_global, v_y_global, v_z_global = gps.getSpeedVector()
        # print(gps.getSpeedVector())

        if it_idx == 0:
            # Initialization
            starting_altitude = z_global
            v_x_global = (x_global - x_global)/dt
            v_y_global = (y_global - y_global)/dt
            v_z_global = (z_global - z_global)/dt
        
        else:
            v_x_global = (x_global - past_x_global)/dt
            v_y_global = (y_global - past_y_global)/dt
            v_z_global = (z_global - past_z_global)/dt

        ########### ------------------ SAVING THINGS -------------------- ###########

        data['GPS'] = gps.getValues()
        data['velocities'] = [v_x_global, v_y_global, v_z_global]
        data['IMU'] = imu.getRollPitchYaw()
        data['gyro'] = gyro.getValues()
        
        ########### ------------------ SAVING THINGS -------------------- ###########

        ## Get body fixed velocities

        rotation_matrix_world_drone = rotation.rotation_matrix(roll, pitch, yaw)
        wd_tr = np.array([x_global, y_global, z_global])
        extrinsic_matrix_world_drone = rotation.get_extrinsic_matrix(rotation_matrix_world_drone, translation_vector=wd_tr)

        v_global = np.array([v_x_global, v_y_global, v_z_global, roll_rate, pitch_rate, yaw_rate])
        twist_world_drone = geometry.velocity_twist_matrix(rotation_matrix_world_drone.T, np.zeros(shape=(3,)))
        v_drone_state = twist_world_drone@v_global
        v_x, v_y, v_z = v_drone_state[:3] # Body velocities 

        # CAMERA IMAGES
        w, h = camera.getWidth(), camera.getHeight()
        cameraData = camera.getImage()  # Note: uint8 string
        image = np.frombuffer(cameraData, np.uint8).reshape(h, w, 4) # BGR, alpha (transparency)
        
        img = img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # gray-scale image
    
        if take_off:
            
            info = tasks['take_off'] # take_off_info
            
            # print(info)

            isclose = np.isclose(z_global, info['setpoints']['position.z'], rtol=1e-2)
            if isclose and prev_step:
                # print(True)
                # print("prev_step", prev_step)
                if prev_step:
                    print('Hovering...')
                    hovering += 1
                    prev_step = isclose
                    if hovering >= hovering_steps:
                        take_off = False
                        visual_servoing = True
                        info['hovering_steps'] = hovering # Save the hovering number of steps
                        info['ending_step'] = it_idx
                        print("Going in front of the gate...")
                        # break
            else:
                hovering = 0
                prev_step = isclose
        
            # print(hovering_steps)
            
            # Setpoints fashion (only in velocity)
            forward_desired = info['setpoints']['velocity.x']
            sideways_desired = info['setpoints']['velocity.y']
            height_desired = info['setpoints']['position.z']
            yaw_desired = info['setpoints']['attitudeRate.yaw']
            
            # New height. Integrate v_z to get the next position.
            height_desired += height_diff_desired * dt 

            ## PID velocity controller with fixed height. Height given as position.
            motor_power = PID_CF.pid(dt, forward_desired, sideways_desired,
                                    yaw_desired, height_desired,
                                    roll, pitch, yaw_rate,
                                    altitude, v_x, v_y, gains)
        
            cv2.imshow("Drone Camera", cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            cv2.waitKey(timestep)

        elif visual_servoing:
            
            info = tasks['visual_servoing']
            
            # print(info)
            
            ########### ------------------ ROTATIONS ------------------ ###########
            rotation_matrix_world_drone = rotation.rotation_matrix(roll, pitch, yaw)
            wd_tr = np.array([x_global, y_global, z_global])
            extrinsic_matrix_world_drone = rotation.get_extrinsic_matrix(rotation_matrix_world_drone, translation_vector=wd_tr)
            
            rotation_matrix_drone_camera = rotation.rotation_matrix(roll=-np.pi/2, pitch=0, yaw=-np.pi/2)
            dc_tr = camera_drone_tr
            extrinsic_matrix_drone_camera = rotation.get_extrinsic_matrix(rotation_matrix_drone_camera, translation_vector=dc_tr)

            extrinsic_matrix = extrinsic_matrix_world_drone@extrinsic_matrix_drone_camera
            ########### ------------------ ROTATIONS ------------------ ###########
                        
            # T_C = SE3(wd_tr)*SE3.RPY(roll,pitch,yaw)*SE3(dc_tr)*SE3.RPY(-np.pi/2, 0, -np.pi/2)
            # p_detected = cam.project_point(P, pose=SE3(T_C, check=False)) 
            p_detected = corner.detect_corners(img)
            # if not vs_init:
            #     # First time, we initialize old_p_detected
            #     current_p_detected = corner.detect_corners(img)
            #     old_p_detected = current_p_detected
            #     vs_init = True
            # else:
            #     current_p_detected = corner.detect_corners(img)
            #     # If none, we consider the 
            #     if current_p_detected is None:
            #         p_detected = old_p_detected
            #     else:
            #         # 1: Consider only the current state; 0: Consider only the previous one
            #         p_detected = corner.weigh_detection(current_p_detected, old_p_detected, alpha=1) 
            #         old_p_detected = p_detected

            ########### ------------------ VISUAL SERVOING ------------------ ###########
            
            # image-plane error
            try:
                
                e = pd - p_detected
                err = np.linalg.norm(e)
                
                print(f"Error: {err:.2f}")
                
                if err <= thresh:
                    
                    visual_servoing = False
                    cross_the_gate = True
                    
                    # Save the current altitude in order to pass the gate
                    # tasks['cross_the_gate']['setpoints']['position.z'] = z_global 
                    # Now set as velocity.z = 0.0

                    info['ending_step'] = it_idx
                    
                    print("Crossing the gate...")
            
            except Exception as e:
                
                print(e)
                
                continue
            
            # stacked image Jacobian
            J = cam.visjac_p(p_detected, Z)
            v_camera = lmda * np.linalg.pinv(J) @ e.T.flatten()

            # Twist velocity from camera frame to drone frame
            twist_drone_camera = geometry.velocity_twist_matrix(rotation_matrix_drone_camera, dc_tr)
            v_drone = twist_drone_camera@v_camera
            ibvs_v_x, ibvs_v_y, ibvs_v_z, ibvs_w_x, ibvs_w_y, ibvs_w_z = v_drone
            
            forward_desired = ibvs_v_x
            sideways_desired = ibvs_v_y
            yaw_desired = ibvs_w_z
            height_diff_desired = ibvs_v_z

            ########### ------------------ VISUAL SERVOING ------------------ ###########
            
            # New height. Integrate v_z to get the next position.
            height_desired += height_diff_desired * dt 
            
            ## PID velocity controller with fixed height. Height given as position.
            motor_power = PID_CF.pid(dt, forward_desired, sideways_desired,
                                    yaw_desired, height_desired,
                                    roll, pitch, yaw_rate,
                                    altitude, v_x, v_y, gains)
            
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

                if collect_data:
                    # Save the image
                    cv2.imwrite(imgs_ibvs_folder+f'/img_{it_idx}.png', cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    
            cv2.imshow("Drone Camera", cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            cv2.waitKey(timestep)
            
            ########### ------------------ SAVING THINGS -------------------- ########### 

            sample = {}
            sample['ibvs_velocities_body_frame'] = [ibvs_v_x, ibvs_v_y, ibvs_v_z, ibvs_w_x, ibvs_w_y, ibvs_w_z]
            sample['target_points'] = pd
            sample['detected_points'] = p_detected
            sample['ibvs_error'] = err
            data['IBVS'] = sample
        
            ########### ------------------ SAVING THINGS -------------------- ###########
    

        elif cross_the_gate:
            
            info = tasks['cross_the_gate']
            
            # print(info)
            
            if cross_the_gate_steps >= info['num_steps']:
                
                hovering += 1

                # Setpoints fashion (only in velocity)
                forward_desired = 0.0
                sideways_desired = 0.0
                height_diff_desired = 0.0
                yaw_desired = info['setpoints']['attitudeRate.yaw']
                
                if hovering >= hovering_steps:
                    cross_the_gate = False
                    info['hovering_steps'] = hovering
                    info['ending_step'] = it_idx
                    landing = True
                    print("Landing...")
            else:
                hovering = 0

                # Setpoints fashion (only in velocity)
                forward_desired = info['setpoints']['velocity.x']
                sideways_desired = info['setpoints']['velocity.y']
                height_diff_desired = info['setpoints']['velocity.z']
                yaw_desired = info['setpoints']['attitudeRate.yaw']
            
            # New height. Integrate v_z to get the next position.
            height_desired += height_diff_desired * dt 

            ## PID velocity controller with fixed height. Height given as position.
            motor_power = PID_CF.pid(dt, forward_desired, sideways_desired,
                                    yaw_desired, height_desired,
                                    roll, pitch, yaw_rate,
                                    altitude, v_x, v_y, gains)
            # print(f'{cross_the_gate_steps}/{info["num_steps"]}')
            cross_the_gate_steps += 1

            cv2.imshow("Drone Camera", cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            cv2.waitKey(timestep)

        elif landing:
            
            info = tasks['land']
            
            # print(info)
            
            # Setpoints fashion (only in velocity)
            forward_desired = info['setpoints']['velocity.x']
            sideways_desired = info['setpoints']['velocity.y']
            height_diff_desired = info['setpoints']['velocity.z']
            
            if landing_steps == 0:
                # Count the number of steps necessary to land when the velocity is height_diff_desired (0.1 m/s)
                num_steps = int(np.ceil((z_global / abs(height_diff_desired)) * sampling_frequency))
                info['num_steps'] = num_steps
            
            yaw_desired = info['setpoints']['attitudeRate.yaw']
            
            # New height. Integrate v_z to get the next position.
            height_desired += height_diff_desired * dt 

            ## PID velocity controller with fixed height. Height given as position.
            motor_power = PID_CF.pid(dt, forward_desired, sideways_desired,
                                    yaw_desired, height_desired,
                                    roll, pitch, yaw_rate,
                                    altitude, v_x, v_y, gains)

            cv2.imshow("Drone Camera", cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            cv2.waitKey(timestep)

            if landing_steps >= num_steps:
                info['ending_step'] = it_idx
                landing = False
                motor_power = [0,0,0,0] # Switch off the motors
                m1_motor.setVelocity(-motor_power[0])
                m2_motor.setVelocity(motor_power[1])
                m3_motor.setVelocity(-motor_power[2])
                m4_motor.setVelocity(motor_power[3])
                print("Landed!")
            
            # print(f'{landing_steps}/{info["num_steps"]}')
            
            landing_steps += 1
        
        else:
            print("No tasks! Controller off.")
            break # No more tasks

        
        m1_motor.setVelocity(-motor_power[0])
        m2_motor.setVelocity(motor_power[1])
        m3_motor.setVelocity(-motor_power[2])
        m4_motor.setVelocity(motor_power[3])

        past_time = robot.getTime()
        past_x_global = x_global
        past_y_global = y_global
        past_z_global = z_global
        
        # ########### ------------------ PRINT -------------------- ########### 
        # print("########### ------------------ GPS [m] -------------------- ###########")
        # print(f"X: {x_global:.4f}\tY: {y_global:.4f}\tZ: {z_global:.4f}")
        # print("########### ------------------ GPS [m] -------------------- ###########")
        # print("\n ")
        # print("########### ------------------ VELOCITIES [m/s] -------------------- ###########")
        # print(f"X: {v_x_global:.4f}\tY: {v_y_global:.4f}\tZ: {v_z_global:.4f}")
        # print("########### ------------------ VELOCITIES [m/s] -------------------- ###########")
        # print("\n ")
        # print("########### ------------------ BODY VELOCITIES [m/s] -------------------- ###########")
        # print(f"X: {v_x:.4f}\tY: {v_y:.4f}\tZ: NO")
        # print("########### ------------------ BODY VELOCITIES [m/s] -------------------- ###########")
        # print("\n ")
        # print("########### ------------------ IMU [rad] -------------------- ###########")
        # print(f"R: {roll:.4f}\tP: {pitch:.4f}\tY: {yaw:.4f}")
        # print("########### ------------------ IMU [rad] -------------------- ###########")
        # print("\n ")
        # print(f"########### ------------------ ATTITUDE RATES [rad/s] -------------------- ###########")
        # print(f"R: {roll_rate:.4f}\tP: {pitch_rate:.4f}\tY: {yaw_rate:.4f}")
        # print(f"########### ------------------ ATTITUDE RATES [rad/s] -------------------- ###########")
        # print("\n ")
        # ########### ------------------ PRINT -------------------- ########### 
        
        ########### ------------------ SAVING THINGS -------------------- ###########

        data['STATE'] = {'POSITION': (x_global, y_global, z_global), 
                        'VELOCITY_GLOBAL': (v_x_global, v_y_global, v_z_global),
                        'VELOCITY_BODY': (v_x, v_y, v_z),
                        'ATTITUDE': (roll, pitch, yaw)}
        data['SENSOR'] = {'ATTITUDE-RATE': (roll_rate, pitch_rate, yaw_rate)}
        data['SETPOINT'] = {'POSITION': (None, None, height_desired),
                            'VELOCITY_BODY': (forward_desired, sideways_desired, height_diff_desired),
                            'ATTITUDE': (None, None, None),
                            'ATTITUDE-RATE': (None, None, yaw_desired)}
        
        data['MOTORS'] = {'m1':-motor_power[0],
                          'm2':motor_power[1],
                          'm3':-motor_power[2],
                          'm4':motor_power[3],}

        if collect_data:
            
            # Save image
            path = f'{imgs_folder}/img_{it_idx}.png'
            cv2.imwrite(path, img) # gray-scale image

        # Save data
        dataset[it_idx] = data

        ########### ------------------ SAVING THINGS -------------------- ###########
        
        it_idx += 1
        
    dataset['info'] = tasks

    import pickle, os

    ########### ------------------ SAVING THINGS -------------------- ###########

    if collect_data:
        
        print("Saving data...")
        with open(folder + '/data.pickle', 'wb') as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Data saved in {folder}.")
    ########### ------------------ SAVING THINGS -------------------- ###########
            
