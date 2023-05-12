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

from math import cos, sin

import numpy as np

import sys
sys.path.append('../../../controllers')
# import os
# print(os.listdir('../../../controllers'))
from pid_controller_new_not_working import pid_velocity_fixed_height_controller

FLYING_ATTITUDE = 1

if __name__ == '__main__':

    robot = Robot()
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

    ## Get keyboard
    keyboard = Keyboard()
    keyboard.enable(timestep)

    ## Initialize variables

    past_x_global = 0
    past_y_global = 0
    past_z_global = 0
    past_time = robot.getTime()

    # Crazyflie velocity PID controller
    PID_CF = pid_velocity_fixed_height_controller()
    PID_update_last_time = robot.getTime()
    sensor_read_last_time = robot.getTime()

    height_desired = FLYING_ATTITUDE
    height_desired = 0

    print("\n");

    print("====== Controls =======\n\n");

    print(" The Crazyflie can be controlled from your keyboard!\n");
    print(" All controllable movement is in body coordinates\n");
    print("- Use the up, back, right and left button to move in the horizontal plane\n");
    print("- Use Q and E to rotate around yaw ");
    print("- Use W and S to go up and down\n ")

    # Capture n steps
    step = 0
    
    # Desired states
    tasks = {}

    take_off = True
    take_off_info = {'setpoints': {'velocity.x':0.0, 'velocity.y':0.0, 'velocity.z':0.1, 'attitudeRate.yaw':0.0}}
    tasks[0] = take_off_info

    first_task = False
    first_task_info = {'setpoints': {'velocity.x':0.1, 'velocity.y':.1, 'velocity.z':.01, 'attitudeRate.yaw':0}, 'num_steps':1000}
    first_task_step = 0
    tasks[1] = first_task_info

    second_task = False

    prev_step = False
    hovering_steps = 0 # Counter used to check how many times we are in the desired state.

    dataset = {'info':tasks, 'sampling_frequency':sampling_frequency}

    ## Initialize values
    desired_state = [0, 0, 0, 0] # Not used
    forward_desired = 0
    sideways_desired = 0
    yaw_desired = 0
    height_diff_desired = 0
    
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
        cosyaw = cos(yaw)
        sinyaw = sin(yaw)
        v_x = v_x_global * cosyaw + v_y_global * sinyaw
        v_y = - v_x_global * sinyaw + v_y_global * cosyaw

        if take_off:
            
            info = tasks[0] # take_off_info
            
            print(info)

            isclose = np.isclose(z_global, 1.0, rtol=1e-2)
            if isclose and prev_step:
                print(True)
                print("prev_step", prev_step)
                if prev_step:
                    hovering_steps += 1
                    prev_step = isclose
                    if hovering_steps >= 500:
                        take_off = False
                        first_task = True
                        print("Passing to the next task...")
                        # break
            else:
                hovering_steps = 0
                prev_step = isclose
        
            print(hovering_steps)
            
            # Setpoints fashion (only in velocity)
            forward_desired = info['setpoints']['velocity.x']
            sideways_desired = info['setpoints']['velocity.y']
            
            if np.isclose(height_desired, 1.0, rtol=1e-2):
                height_diff_desired = 0.0
            else:   
                height_diff_desired = info['setpoints']['velocity.z']
            print('height_diff_desired',height_diff_desired)
            yaw_desired = info['setpoints']['attitudeRate.yaw']
            
            # New height. Integrate v_z to get the next position.
            height_desired += height_diff_desired * dt 

            ## Example how to get sensor data
            ## range_front_value = range_front.getValue();
            ## cameraData = camera.getImage()


            ## PID velocity controller with fixed height. Height given as position.
            motor_power = PID_CF.pid(dt, forward_desired, sideways_desired,
                                    yaw_desired, height_desired,
                                    roll, pitch, yaw_rate,
                                    altitude, v_x, v_y)
        
        elif first_task:
            
            info = tasks[1]
            
            print(info)
            
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
                                    altitude, v_x, v_y)
        
        # print(motor_power)
        
        m1_motor.setVelocity(-motor_power[0])
        m2_motor.setVelocity(motor_power[1])
        m3_motor.setVelocity(-motor_power[2])
        m4_motor.setVelocity(motor_power[3])

        past_time = robot.getTime()
        past_x_global = x_global
        past_y_global = y_global
        past_z_global = z_global
        
        ########### ------------------ PRINT -------------------- ########### 
        print("########### ------------------ GPS [m] -------------------- ###########")
        print(f"X: {x_global:.4f}\tY: {y_global:.4f}\tZ: {z_global:.4f}")
        print("########### ------------------ GPS [m] -------------------- ###########")
        print("\n ")
        print("########### ------------------ VELOCITIES [m/s] -------------------- ###########")
        print(f"X: {v_x_global:.4f}\tY: {v_y_global:.4f}\tZ: {v_z_global:.4f}")
        print("########### ------------------ VELOCITIES [m/s] -------------------- ###########")
        print("\n ")
        # print("########### ------------------ BODY VELOCITIES [m/s] -------------------- ###########")
        # print(f"X: {v_x:.4f}\tY: {v_y:.4f}\tZ: NO")
        # print("########### ------------------ BODY VELOCITIES [m/s] -------------------- ###########")
        # print("\n ")
        print("########### ------------------ IMU [rad] -------------------- ###########")
        print(f"R: {roll:.4f}\tP: {pitch:.4f}\tY: {yaw:.4f}")
        print("########### ------------------ IMU [rad] -------------------- ###########")
        print("\n ")
        print(f"########### ------------------ ATTITUDE RATES [rad/s] -------------------- ###########")
        print(f"R: {roll_rate:.4f}\tP: {pitch_rate:.4f}\tY: {yaw_rate:.4f}")
        print(f"########### ------------------ ATTITUDE RATES [rad/s] -------------------- ###########")
        print("\n ")
        ########### ------------------ PRINT -------------------- ########### 
        
        ########### ------------------ SAVING THINGS -------------------- ###########

        data['STATE'] = {'POSITION': (x_global, y_global, z_global), 
                        'VELOCITY': (v_x_global, v_y_global, v_z_global),
                        'ATTITUDE': (roll, pitch, yaw)}
        data['SENSOR'] = {'ATTITUDE-RATE': (roll_rate, pitch_rate, yaw_rate)}
        data['SETPOINT'] = {'POSITION': (None, None, height_desired),
                            'VELOCITY': (forward_desired, sideways_desired, height_diff_desired),
                            'ATTITUDE': (None, None, None),
                            'ATTITUDE-RATE': (None, None, yaw_desired)}
        
        # Save data
        dataset[step] = data

        ########### ------------------ SAVING THINGS -------------------- ###########
        
        step += 1
        
    import pickle, os

    # Set to True if you want to collect data
    collect_data = False

    parent_folder = '../../datasets/EXP-3-CRAZYFLIE-CONTROLLERS-TEST'
    folder = parent_folder +'/tests'+ '/01_controller_py_test'

    if not os.path.isdir(folder):
        os.makedirs(folder)

    ########### ------------------ SAVING THINGS -------------------- ###########
    if collect_data:
        print("Saving data...")
        with open(folder + '/data.pickle', 'wb') as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Data saved in {folder}.")
    ########### ------------------ SAVING THINGS -------------------- ###########
            