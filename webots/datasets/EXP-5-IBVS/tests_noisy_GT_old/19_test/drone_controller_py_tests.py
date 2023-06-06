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

import sys, shutil
sys.path.append('../../../controllers/')

# Import the path for the corner detection module
sys.path.append('../../../../scarciglia-nanodrone-gate-detection/')

from src import corner, rotation, geometry
from pid_controller import pid_velocity_fixed_height_controller

FLYING_ATTITUDE = 1
np.random.seed(0)

########### ------------------ TEST PARAMETERS -------------------- ###########
relative_angle = np.deg2rad(-45)
test = {# 'orientation':(0,0,np.pi/2), # Perpendicular to the gate
        'orientation':(0,0,np.pi/2 + relative_angle), # Facing the gate centre
        'theta':relative_angle, 
        'take_off_height':0.5, 
        'sigma':0.437}
########### ------------------ TEST PARAMETERS -------------------- ###########

########### ------------------ SAVING THINGS -------------------- ###########
    
# Set to True if you want to collect data
collect_data = False

if collect_data:
        
    parent_folder = '../../datasets/EXP-5-IBVS'
    folder = parent_folder +'/tests_noisy_GT/'+ '19_test'


    imgs_folder = f'{folder}/imgs/'
    imgs_ibvs_folder = f'{folder}/imgs_ibvs/'
    # contours_folder = f'{imgs_ibvs_folder}' + 'contours/'

    try:
        if os.path.isdir(imgs_folder):
            shutil.rmtree(imgs_folder)
        os.makedirs(imgs_folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..
        else:
            print(f"Overwriting folder {imgs_folder}")

    try:
        if os.path.isdir(imgs_ibvs_folder):
            shutil.rmtree(imgs_ibvs_folder)
        os.makedirs(imgs_ibvs_folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..
        else:
            print(f"Overwriting folder {imgs_ibvs_folder}")

    # try:
    #     if os.path.isdir(contours_folder):
    #         shutil.rmtree(contours_folder)
    #     os.makedirs(contours_folder)
    # except OSError as e:
    #     if e.errno != errno.EEXIST:
    #         raise  # This was not a "directory exist" error..
    #     else:
    #         print(f"Overwriting folder {contours_folder}")
    
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

    ########### ------------------ DRONE STARTING POSITION ------------------ ###########
    r = 2
    theta = test['theta']

    x = r*np.sin(theta)
    y = - r*np.cos(theta)

    starting_height = translation_drone.getSFVec3f()[-1]
    translation_drone.setSFVec3f([x, y, starting_height])

    # Orientation
    roll, pitch, yaw = test['orientation']
    rot = transforms3d.euler.euler2axangle(roll, -pitch, yaw)
    ax_angle = list(rot[0])
    ax_angle.append(rot[-1])    

    crazyflie_node.getField('rotation').setSFRotation(ax_angle)
    ########### ------------------ DRONE STARTING POSITION ------------------ ###########


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
    take_off_info = {'setpoints': {'velocity.x':0.0, 'velocity.y':0.0, 'position.z':test['take_off_height'], 'attitudeRate.yaw':0.0}}
    tasks['take_off'] = take_off_info

    visual_servoing = False
    # old_p_detected = None
    detection = np.zeros(shape=(3,2,4))
    GT_detection = np.zeros(shape=(3,2,4))
    detections = []
    GT_detections = []
    GT_jitter_detections = []
    filter = {'alpha':1, 'order':1}
    vs_counter = 0
    track_error = False
    offset = None
    errors = np.zeros(shape=(3*sampling_frequency))
    median_err = np.inf
    tasks['visual_servoing'] = {'visual_servoing':True, 'corner_detection':True, 'filter':filter, 'next_task_condition':{'median_error':median_err, 'error_array':errors}}
    
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
    thresh = 8
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
    dataset['test'] = test

    ########### ------------------ SAVING THINGS ------------------ ###########

    height_desired = take_off_info['setpoints']['position.z']

    ## Initialize values
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

        ## Get sensor data
        roll, pitch, yaw = imu.getRollPitchYaw()
        roll_rate, pitch_rate, yaw_rate = gyro.getValues()
        altitude = gps.getValues()[2]
        x_global, y_global, z_global = gps.getValues()


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
            
            isclose = np.isclose(z_global, info['setpoints']['position.z'], rtol=1e-2)
            if isclose and prev_step:
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
                        
            ########### ------------------ ROTATIONS ------------------ ###########
            rotation_matrix_world_drone = rotation.rotation_matrix(roll, pitch, yaw)
            wd_tr = np.array([x_global, y_global, z_global])
            extrinsic_matrix_world_drone = rotation.get_extrinsic_matrix(rotation_matrix_world_drone, translation_vector=wd_tr)
            
            rotation_matrix_drone_camera = rotation.rotation_matrix(roll=-np.pi/2, pitch=0, yaw=-np.pi/2)
            dc_tr = camera_drone_tr
            extrinsic_matrix_drone_camera = rotation.get_extrinsic_matrix(rotation_matrix_drone_camera, translation_vector=dc_tr)

            extrinsic_matrix = extrinsic_matrix_world_drone@extrinsic_matrix_drone_camera
            ########### ------------------ ROTATIONS ------------------ ###########
            
            ########### ------------------ GT VISUAL SERVOING ------------------ ###########
                       
            T_C = SE3(wd_tr)*SE3.RPY(roll,pitch,yaw)*SE3(dc_tr)*SE3.RPY(-np.pi/2, 0, -np.pi/2)
            GT_p_detected = cam.project_point(P, pose=SE3(T_C, check=False)) 
            current_p_detected = GT_p_detected
            if vs_counter == 0:
                GT_detection[0] = current_p_detected
                GT_p_detected = current_p_detected
            elif vs_counter == 1:
                GT_detection[1] = GT_detection[0]
                GT_detection[0] = current_p_detected
                GT_p_detected = corner.weigh_detection(GT_detection, order=1, alpha=filter['alpha'])
            else:
                GT_detection[2] = GT_detection[1]
                GT_detection[1] = GT_detection[0]
                GT_detection[0] = current_p_detected
                GT_p_detected = corner.weigh_detection(GT_detection, order=filter['order'], alpha=filter['alpha'])
                       
            GT_detections.append(GT_p_detected)
                        
            jitter = np.random.normal(scale=test['sigma'], size=8).reshape(2,4) 
            
            GT_jitter_p_detected = GT_p_detected + jitter
            
            GT_jitter_detections.append(GT_jitter_p_detected)
            
            # image-plane error
            try:
                
                GT_e = pd - GT_jitter_p_detected
                GT_err = np.linalg.norm(GT_e)
                
                print(f"Error: {GT_err:.2f}")

                if GT_err <= 50 and track_error is False:
                    track_error = True
                    offset = it_idx
                    print("Collect errors...")
                
                if track_error and median_err > 50:
                    idx = it_idx - offset
                    if idx >= len(errors):
                        idx = len(errors)
                    print(f"Collecting errors: {idx}/{errors.shape[-1]}")                    
                    if idx == len(errors):
                        median_err = np.median(errors)
                        print("Median error:", median_err)
                        if median_err <= 50:
                            visual_servoing = False
                            cross_the_gate = True
                            
                            # Save the current altitude in order to pass the gate
                            # tasks['cross_the_gate']['setpoints']['position.z'] = z_global 
                            # Now set as velocity.z = 0.0

                            info['ending_step'] = it_idx
                            
                            print("Crossing the gate...")
                        else:
                            # Shift errors
                            temp = errors[1:]
                            errors[:-1] = temp
                            errors[-1] = GT_err
                    else:
                        errors[idx] = GT_err

                if GT_err <= thresh:
                    
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
            J = cam.visjac_p(GT_jitter_p_detected, Z)
            v_camera = lmda * np.linalg.pinv(J) @ GT_e.T.flatten()
   
            # Twist velocity from camera frame to drone frame
            twist_drone_camera = geometry.velocity_twist_matrix(rotation_matrix_drone_camera, dc_tr)
            
            v_drone = twist_drone_camera@v_camera
            GT_ibvs_v_x, GT_ibvs_v_y, GT_ibvs_v_z, GT_ibvs_w_x, GT_ibvs_w_y, GT_ibvs_w_z = v_drone
            
            forward_desired = GT_ibvs_v_x
            sideways_desired = GT_ibvs_v_y
            yaw_desired = GT_ibvs_w_z
            height_diff_desired = GT_ibvs_v_z
            
            # New height. Integrate v_z to get the next position.
            height_desired += height_diff_desired * dt 

            ########### ------------------ GT VISUAL SERVOING ------------------ ##########

            # Show image
            for id, col in enumerate(colors):
                tl = pd[:,0] # 0
                bl = pd[:,1] # 1
                br = pd[:,2] # 2
                tr = pd[:,3] # 3 
                x, y = pd[:,id] # Desired
                image = cv2.putText(image, text=str(id), org = (int(x),int(y)), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.5, color = (255,255,255), thickness = 1)
                image = cv2.circle(image, (int(x),int(y)), radius=2, color=(255, 255, 255), thickness=1)
                image = cv2.line(image, (int(tl[0]), int(tl[1])), (int(tr[0]), int(tr[1])), color=(255, 255, 255), thickness=1) # top-left, top-right
                image = cv2.line(image, (int(tr[0]), int(tr[1])), (int(br[0]), int(br[1])), color=(255, 255, 255), thickness=1) # top-right, bottom-right
                image = cv2.line(image, (int(br[0]), int(br[1])), (int(bl[0]), int(bl[1])), color=(255, 255, 255), thickness=1) # bottom-left, top-right
                image = cv2.line(image, (int(bl[0]), int(bl[1])), (int(tl[0]), int(tl[1])), color=(255, 255, 255), thickness=1) # bottom-left, top-left
                x, y = GT_p_detected[:,id] # Detected
                image = cv2.circle(image, (int(x),int(y)), radius=2, color=(255, 255, 255), thickness=-1)
                image = cv2.putText(image, text=str(id), org = (int(x),int(y)), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.5, color = (255,255,255), thickness = 1)
                if collect_data:
                    # Save the image
                    cv2.imwrite(imgs_ibvs_folder+f'/img_{it_idx}.png', cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    
            cv2.imshow("Drone Camera", cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            cv2.waitKey(timestep)
            
            ########### ------------------ DETECTION VISUAL SERVOING ------------------ ###########

            current_p_detected = corner.detect_corners(img, return_drawing=False)

            if vs_counter == 0:
                detection[0] = current_p_detected
                p_detected = current_p_detected
            elif vs_counter == 1:
                detection[1] = detection[0]
                detection[0] = current_p_detected
                p_detected = corner.weigh_detection(detection, order=1, alpha=filter['alpha'])
            else:
                detection[2] = detection[1]
                detection[1] = detection[0]
                detection[0] = current_p_detected
                p_detected = corner.weigh_detection(detection, order=filter['order'], alpha=filter['alpha'])

            vs_counter += 1

            detections.append(p_detected)

            # image-plane error
            try:
                
                e = pd - p_detected
                err = np.linalg.norm(e)
                
            except Exception as e:
                
                print(e)
                
                continue
            
            try:
                # stacked image Jacobian
                J = cam.visjac_p(p_detected, Z)
                v_camera = lmda * np.linalg.pinv(J) @ e.T.flatten()
                # Twist velocity from camera frame to drone frame
                twist_drone_camera = geometry.velocity_twist_matrix(rotation_matrix_drone_camera, dc_tr)
                v_drone = twist_drone_camera@v_camera
            
            except Exception as e:

                print(e)    
                
                info['ending_step'] = it_idx
                
                ibvs_v_x, ibvs_v_y, ibvs_v_z, ibvs_w_x, ibvs_w_y, ibvs_w_z = np.full(shape=(6,), fill_value=np.nan)
                
                ########### ------------------ SAVING THINGS -------------------- ########### 

                sample = {}
                sample['ibvs_velocities_body_frame'] = [ibvs_v_x, ibvs_v_y, ibvs_v_z, ibvs_w_x, ibvs_w_y, ibvs_w_z]
                sample['GT_ibvs_velocities_body_frame'] = [GT_ibvs_v_x, GT_ibvs_v_y, GT_ibvs_v_z, GT_ibvs_w_x, GT_ibvs_w_y, GT_ibvs_w_z]
                sample['target_points'] = pd
                sample['detected_points'] = p_detected
                sample['GT_detected_points'] = GT_p_detected
                sample['GT_jitter_detected_points'] = GT_jitter_p_detected
                sample['ibvs_error'] = err
                sample['GT_jitter_ibvs_error'] = GT_err
                data['IBVS'] = sample
            
                ########### ------------------ SAVING THINGS -------------------- ###########
                
                break
                
            ibvs_v_x, ibvs_v_y, ibvs_v_z, ibvs_w_x, ibvs_w_y, ibvs_w_z = v_drone

            ########### ------------------ DETECTION VISUAL SERVOING ------------------ ###########
            
            ## PID velocity controller with fixed height. Height given as position.
            motor_power = PID_CF.pid(dt, forward_desired, sideways_desired,
                                    yaw_desired, height_desired,
                                    roll, pitch, yaw_rate,
                                    altitude, v_x, v_y, gains)
            
            ########### ------------------ SAVING THINGS -------------------- ########### 

            sample = {}
            sample['ibvs_velocities_body_frame'] = [ibvs_v_x, ibvs_v_y, ibvs_v_z, ibvs_w_x, ibvs_w_y, ibvs_w_z]
            sample['GT_ibvs_velocities_body_frame'] = [GT_ibvs_v_x, GT_ibvs_v_y, GT_ibvs_v_z, GT_ibvs_w_x, GT_ibvs_w_y, GT_ibvs_w_z]
            sample['target_points'] = pd
            sample['detected_points'] = p_detected
            sample['GT_detected_points'] = GT_p_detected
            sample['GT_jitter_detected_points'] = GT_jitter_p_detected
            sample['ibvs_error'] = err
            sample['GT_jitter_ibvs_error'] = GT_err
            data['IBVS'] = sample
            
            ########### ------------------ SAVING THINGS -------------------- ###########
                

        elif cross_the_gate:
            
            info = tasks['cross_the_gate']
                        
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
    dataset['detections'] = {'detection': detections, 'GT_detection': GT_detections, 'GT_jitter_detection': GT_jitter_detections}
    dataset['jitter'] = {'mean':0, 'std_dev':0.45}

    import pickle, os

    ########### ------------------ SAVING THINGS -------------------- ###########

    if collect_data:
        
        print("Saving data...")
        with open(folder + '/data.pickle', 'wb') as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Data saved in {folder}.")
    ########### ------------------ SAVING THINGS -------------------- ###########
            

