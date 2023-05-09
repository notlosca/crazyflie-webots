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

from math import cos, sin, degrees, radians 

import sys
# Change this path to your crazyflie-firmware folder
sys.path.append('../../../../crazyflie-firmware/build')
import cffirmware

import numpy as np

robot = Robot()

timestep = int(robot.getBasicTimeStep())

## Initialize motors
m1_motor = robot.getDevice("m1_motor")
m1_motor.setPosition(float('inf'))
m1_motor.setVelocity(-1)
m2_motor = robot.getDevice("m2_motor")
m2_motor.setPosition(float('inf'))
m2_motor.setVelocity(1)
m3_motor = robot.getDevice("m3_motor")
m3_motor.setPosition(float('inf'))
m3_motor.setVelocity(-1)
m4_motor = robot.getDevice("m4_motor")
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
pastXGlobal = 0
pastYGlobal = 0
pastZGlobal = 0

past_time = robot.getTime()

cffirmware.controllerPidInit()

print('Take off!')

# Capture 10000 steps
step = 0
# dataset = {'info':{'setpoint':'velocity', 'vx':0.1, 'vy':0.1, 'vz':0.1}}
# dataset = {'info':{'setpoint':'attitude_rate (rad/s)', 'roll':8, 'pitch':8, 'yaw':8, 'z':1.0}}

# Desired states
x_desired = 0.0
y_desired = 0.0
z_desired = 1.0
first_task = True
second_task = False

dataset = {'info':{'setpoint':'position', 'x':x_desired, 'y':y_desired, 'z':z_desired, 'yaw':'disabled'}}

positions = np.zeros(shape=(11,3))

vals = np.linspace(0,1,11,endpoint=True)

positions[:,-1] = vals

temp = np.zeros(shape=(10,3))
temp[:,1] = vals[1:]
temp[:,-1] = vals[-1]
positions = np.vstack((positions, temp))

temp = np.zeros(shape=(10,3))
temp[:,0] = vals[1:]
temp[:,1] = vals[-1]
temp[:,2] = vals[-1]
positions = np.vstack((positions, temp))

cnt = 1

hover_state = np.array([0,0,0])

wait = False
wait_time = 1000/2
wait_step = 0

# Main loop:
while robot.step(timestep) != -1 and step < 5000:
    print(step)
    data = {}
    
    dt = robot.getTime() - past_time

    ## Get measurements
    roll = imu.getRollPitchYaw()[0]
    pitch = imu.getRollPitchYaw()[1]
    yaw = imu.getRollPitchYaw()[2]
    roll_rate = gyro.getValues()[0]
    pitch_rate = gyro.getValues()[1]
    yaw_rate = gyro.getValues()[2]
    xGlobal = gps.getValues()[0]
    vxGlobal = (xGlobal - pastXGlobal)/dt
    yGlobal = gps.getValues()[1]
    vyGlobal = (yGlobal - pastYGlobal)/dt
    zGlobal = gps.getValues()[2]
    vzGlobal = (zGlobal - pastZGlobal)/dt
    
    ########### ------------------ SAVING THINGS -------------------- ###########

    data['GPS'] = gps.getValues()
    data['velocities'] = [vxGlobal, vyGlobal, vzGlobal]
    data['IMU'] = imu.getRollPitchYaw()
    data['gyro'] = gyro.getValues()
    
    ########### ------------------ SAVING THINGS -------------------- ###########

    ## Put measurement in state estimate
    # TODO replace these with a EKF python binding
    state = cffirmware.state_t()
    state.attitude.roll = degrees(roll)
    state.attitude.pitch = -degrees(pitch)
    state.attitude.yaw = degrees(yaw)
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

    # # keyboard input
    # forwardDesired = 0
    # sidewaysDesired = 0
    # yawDesired = 0
    #
    # key = keyboard.getKey()
    # while key>0:
    #     if key == Keyboard.UP:
    #         forwardDesired = 0.5
    #     elif key == Keyboard.DOWN:
    #         forwardDesired = -0.5
    #     elif key == Keyboard.RIGHT:
    #         sidewaysDesired = -0.5
    #     elif key == Keyboard.LEFT:
    #         sidewaysDesired = 0.5
    #     elif key == ord('Q'):
    #         yawDesired = 8
    #     elif key == ord('E'):
    #         yawDesired = -8
    # 
    #     key = keyboard.getKey()

    ## Example how to get sensor data
    # range_front_value = range_front.getValue();
    # cameraData = camera.getImage()

    # if wait:
    #     if wait_time == wait_step:
    #         wait = False
    #         wait_step = 0
    #     else:
    #         print(wait_step)
    #         ## Fill in Setpoints
    #         # Setpoint mode
    #         setpoint = cffirmware.setpoint_t()
    #         # X, Y, Z
    #         setpoint.mode.x = cffirmware.modeAbs
    #         setpoint.mode.y = cffirmware.modeAbs
    #         setpoint.mode.z = cffirmware.modeAbs
    #         # setpoint.mode.x = cffirmware.modeVelocity
    #         # setpoint.mode.y = cffirmware.modeVelocity
    #         # setpoint.mode.z = cffirmware.modeVelocity
    #         # # R, P, Y
    #         # setpoint.mode.roll = cffirmware.modeAbs
    #         # setpoint.mode.pitch = cffirmware.modeAbs
    #         setpoint.mode.yaw = cffirmware.modeAbs
    #         # setpoint.mode.roll = cffirmware.modeVelocity
    #         # setpoint.mode.pitch = cffirmware.modeVelocity
    #         # setpoint.mode.yaw = cffirmware.modeVelocity
    #         # Position
    #         setpoint.position.x = hover_state[0]
    #         setpoint.position.y = hover_state[1]
    #         setpoint.position.z = hover_state[2]
    #         # # Velocity
    #         # setpoint.velocity.x = 0
    #         # setpoint.velocity.y = 0
    #         # setpoint.velocity.z = 0.1
    #         # Attitude
    #         # setpoint.attitude.roll = degrees(np.pi/6)
    #         # setpoint.attitude.pitch = degrees(np.pi/6)
    #         setpoint.attitude.yaw = 0
    #         # # Attitude rate
    #         # setpoint.attitudeRate.roll = degrees(8)
    #         # setpoint.attitudeRate.pitch = degrees(8)
    #         my_vel = 0.0035
    #         magic_constant = 0.0626
    #         # setpoint.attitudeRate.yaw = degrees(my_vel/magic_constant)
    #         
    #         setpoint.velocity_body = True
# 
    #         ## Firmware PID bindings
    #         control = cffirmware.control_t()
    #         tick = 100 #this value makes sure that the position controller and attitude controller are always always initiated
    #         cffirmware.controllerPid(control, setpoint,sensors,state,tick)
# 
    #         ## 
    #         cmd_roll = radians(control.roll)
    #         cmd_pitch = radians(control.pitch)
    #         cmd_yaw = -radians(control.yaw)
    #         cmd_thrust = control.thrust
# 
    #         ## Motor mixing
    #         motorPower_m1 =  cmd_thrust - cmd_roll + cmd_pitch + cmd_yaw
    #         motorPower_m2 =  cmd_thrust - cmd_roll - cmd_pitch - cmd_yaw
    #         motorPower_m3 =  cmd_thrust + cmd_roll - cmd_pitch + cmd_yaw
    #         motorPower_m4 =  cmd_thrust + cmd_roll + cmd_pitch - cmd_yaw
# 
    #         scaling = 1000 ##Todo, remove necessity of this scaling (SI units in firmware)
    #         m1_motor.setVelocity(-motorPower_m1/scaling)
    #         m2_motor.setVelocity(motorPower_m2/scaling)
    #         m3_motor.setVelocity(-motorPower_m3/scaling)
    #         m4_motor.setVelocity(motorPower_m4/scaling)
    #         
    #         past_time = robot.getTime()
    #         pastXGlobal = xGlobal
    #         pastYGlobal = yGlobal
    #         pastZGlobal = zGlobal
    #         
    #         wait_step += 1
# 
    #         continue
    # 
    
    # x_desired, y_desired, z_desired = positions[cnt]
    
    # x_desired, y_desired, z_desired = [.1,.1,.1]
    
    print("Desired state", [x_desired, y_desired, z_desired])
    
    ## Fill in Setpoints
    # Setpoint mode
    setpoint = cffirmware.setpoint_t()
    # X, Y, Z
    setpoint.mode.x = cffirmware.modeAbs
    setpoint.mode.y = cffirmware.modeAbs
    setpoint.mode.z = cffirmware.modeAbs
    # setpoint.mode.x = cffirmware.modeVelocity
    # setpoint.mode.y = cffirmware.modeVelocity
    # setpoint.mode.z = cffirmware.modeVelocity
    # # R, P, Y
    # setpoint.mode.roll = cffirmware.modeAbs
    # setpoint.mode.pitch = cffirmware.modeAbs
    setpoint.mode.yaw = cffirmware.modeAbs
    # setpoint.mode.roll = cffirmware.modeVelocity
    # setpoint.mode.pitch = cffirmware.modeVelocity
    # setpoint.mode.yaw = cffirmware.modeVelocity
    # Position
    setpoint.position.x = x_desired
    setpoint.position.y = y_desired
    setpoint.position.z = z_desired
    # # Velocity
    # setpoint.velocity.x = 0
    # setpoint.velocity.y = 0
    # setpoint.velocity.z = 0.1
    # Attitude
    # setpoint.attitude.roll = degrees(np.pi/6)
    # setpoint.attitude.pitch = degrees(np.pi/6)
    setpoint.attitude.yaw = 0
    # # Attitude rate
    # setpoint.attitudeRate.roll = degrees(8)
    # setpoint.attitudeRate.pitch = degrees(8)
    my_vel = 0.0035
    magic_constant = 0.0626
    # setpoint.attitudeRate.yaw = degrees(my_vel/magic_constant)
    
    setpoint.velocity_body = True
    
    if cnt != len(positions) - 1:
        if np.isclose(x_desired, xGlobal, rtol=1e-1) and np.isclose(y_desired, yGlobal, rtol=1e-1) and np.isclose(z_desired, zGlobal, rtol=1e-1):
            wait = True
            hover_state = positions[cnt]
            # first_task = False
            # second_task = True
            cnt += 1
    
    
    # ########### ------------------ FIRST TASK -------------------- ########### 
    # if first_task:
    #     ## Fill in Setpoints
    #     # Setpoint mode
    #     setpoint = cffirmware.setpoint_t()
    #     # X, Y, Z
    #     # setpoint.mode.x = cffirmware.modeAbs
    #     # setpoint.mode.y = cffirmware.modeAbs
    #     setpoint.mode.z = cffirmware.modeAbs
    #     setpoint.mode.x = cffirmware.modeVelocity
    #     setpoint.mode.y = cffirmware.modeVelocity
    #     # setpoint.mode.z = cffirmware.modeVelocity
    #     # # R, P, Y
    #     # setpoint.mode.roll = cffirmware.modeAbs
    #     # setpoint.mode.pitch = cffirmware.modeAbs
    #     setpoint.mode.yaw = cffirmware.modeAbs
    #     # setpoint.mode.roll = cffirmware.modeVelocity
    #     # setpoint.mode.pitch = cffirmware.modeVelocity
    #     # setpoint.mode.yaw = cffirmware.modeVelocity
    #     # Position
    #     # setpoint.position.x = 1.0
    #     # setpoint.position.y = 1.0
    #     setpoint.position.z = 1.0
    #     # Velocity
    #     setpoint.velocity.x = 0
    #     setpoint.velocity.y = 0
    #     # setpoint.velocity.z = 0.1
    #     # Attitude
    #     # setpoint.attitude.roll = degrees(np.pi/6)
    #     # setpoint.attitude.pitch = degrees(np.pi/6)
    #     setpoint.attitude.yaw = 0
    #     # Attitude rate
    #     # setpoint.attitudeRate.roll = degrees(8)
    #     # setpoint.attitudeRate.pitch = degrees(8)
    #     my_vel = 0.035
    #     magic_constant = 0.0626
    #     # setpoint.attitudeRate.yaw = degrees(my_vel/magic_constant)
    #     
    #     setpoint.velocity_body = True
    #     
    #     if np.isclose(zGlobal, z_desired, rtol=1e-3):
    #         first_task = False
    #         second_task = True
    # 
    # ########### ------------------ FIRST TASK -------------------- ########### 
# 
    # ########### ------------------ SECOND TASK -------------------- ########### 
    # if second_task:
    #     
    #     ## Fill in Setpoints
    #     # Setpoint mode
    #     setpoint = cffirmware.setpoint_t()
    #     # X, Y, Z
    #     # setpoint.mode.x = cffirmware.modeAbs
    #     # setpoint.mode.y = cffirmware.modeAbs
    #     setpoint.mode.z = cffirmware.modeAbs
    #     setpoint.mode.x = cffirmware.modeVelocity
    #     setpoint.mode.y = cffirmware.modeVelocity
    #     # setpoint.mode.z = cffirmware.modeVelocity
    #     # R, P, Y
    #     # setpoint.mode.roll = cffirmware.modeAbs
    #     # setpoint.mode.pitch = cffirmware.modeAbs
    #     # setpoint.mode.yaw = cffirmware.modeAbs
    #     # setpoint.mode.roll = cffirmware.modeVelocity
    #     # setpoint.mode.pitch = cffirmware.modeVelocity
    #     # setpoint.mode.yaw = cffirmware.modeVelocity
    #     # # Position
    #     # setpoint.position.x = 1.0
    #     # setpoint.position.y = 1.0
    #     setpoint.position.z = 1.0
    #     # Velocity
    #     setpoint.velocity.x = 0.1 + (np.random.randn(1)/100)[0]
    #     setpoint.velocity.y = 0.1 + (np.random.randn(1)/100)[0]
    #     # setpoint.velocity.z = 0.1 + (np.random.randn(1)/10)[0]
    #     # Attitude
    #     # setpoint.attitude.roll = degrees(np.pi/6)
    #     # setpoint.attitude.pitch = degrees(np.pi/6)
    #     # setpoint.attitude.yaw = 90
    #     # Attitude rate
    #     # setpoint.attitudeRate.roll = degrees(8)
    #     # setpoint.attitudeRate.pitch = degrees(8)
    #     my_vel = 0.035
    #     magic_constant = 0.0626
    #     # setpoint.attitudeRate.yaw = degrees(my_vel/magic_constant)
    #     
    #     setpoint.velocity_body = True
    #     
    #     if np.isclose(zGlobal, z_desired) and np.isclose(yGlobal, y_desired):
    #         first_task = False
    #         second_task = False
    #         third_task = True
    # ########### ------------------ SECOND TASK -------------------- ########### 
    
    
        
    ## Firmware PID bindings
    control = cffirmware.control_t()
    tick = 100 #this value makes sure that the position controller and attitude controller are always always initiated
    cffirmware.controllerPid(control, setpoint,sensors,state,tick)

    ## 
    cmd_roll = radians(control.roll)
    cmd_pitch = radians(control.pitch)
    cmd_yaw = -radians(control.yaw)
    cmd_thrust = control.thrust
    
    print('cmd_roll', cmd_roll)
    print('cmd_pitch', cmd_pitch)
    print('cmd_yaw', cmd_yaw)
    print('cmd_thrust', cmd_thrust)

    ## Motor mixing
    motorPower_m1 =  cmd_thrust - cmd_roll + cmd_pitch + cmd_yaw
    motorPower_m2 =  cmd_thrust - cmd_roll - cmd_pitch - cmd_yaw
    motorPower_m3 =  cmd_thrust + cmd_roll - cmd_pitch + cmd_yaw
    motorPower_m4 =  cmd_thrust + cmd_roll + cmd_pitch - cmd_yaw

    print('motorPower_m1', motorPower_m1)
    print('motorPower_m2', motorPower_m2)
    print('motorPower_m3', motorPower_m3)
    print('motorPower_m4', motorPower_m4)

    scaling = 1000 ##Todo, remove necessity of this scaling (SI units in firmware)
    m1_motor.setVelocity(-motorPower_m1/scaling)
    m2_motor.setVelocity(motorPower_m2/scaling)
    m3_motor.setVelocity(-motorPower_m3/scaling)
    m4_motor.setVelocity(motorPower_m4/scaling)
    
    past_time = robot.getTime()
    pastXGlobal = xGlobal
    pastYGlobal = yGlobal
    pastZGlobal = zGlobal
    
    # ########### ------------------ PRINT -------------------- ########### 
    # print("########### ------------------ GPS [m] -------------------- ###########")
    # print(f"X: {xGlobal:.4f}\tY: {yGlobal:.4f}\tZ: {zGlobal:.4f}")
    # print("########### ------------------ GPS [m] -------------------- ###########")
    # print("\n ")
    # print("########### ------------------ VELOCITIES [m/s] -------------------- ###########")
    # print(f"X: {vxGlobal:.4f}\tY: {vyGlobal:.4f}\tZ: {vzGlobal:.4f}")
    # print("########### ------------------ VELOCITIES [m/s] -------------------- ###########")
    # print("\n ")
    # print("########### ------------------ IMU [rad] -------------------- ###########")
    # print(f"R: {roll:.4f}\tP: {pitch:.4f}\tY: {yaw:.4f}")
    # print("########### ------------------ IMU [rad] -------------------- ###########")
    # print("\n ")
    # print(f"########### ------------------ ATTITUDE RATES [rad/s] -------------------- ###########")
    # print(f"R: {roll_rate:.4f}\tP: {pitch_rate:.4f}\tY: {yaw_rate:.4f}")
    # print(f"########### ------------------ ATTITUDE RATES [rad/s] -------------------- ###########")
    # print("\n ")
    # print(f"########### ------------------ YAW COMMAND -------------------- ###########")
    # print(f'Yaw command [rad]:{cmd_yaw:.4f}', f'Control yaw [deg]:{control.yaw}')
    # print(f"########### ------------------ YAW COMMAND -------------------- ###########")
    # ########### ------------------ PRINT -------------------- ########### 
    
    # # ########### ------------------ PRINT STATE -------------------- ########### 
    # 
    # print("[STATE][POSITION] x, y, z", (state.position.x, state.position.y, state.position.z)) # Position
    # print("[STATE][VELOCITY] vx, vy, vz", (state.velocity.x, state.velocity.y, state.velocity.z)) # Velocity
    # print("[STATE][ATTITUDE] r, p, y", (state.attitude.roll, state.attitude.pitch, state.attitude.yaw)) # Attitude
    # print("[SENSOR][ATTITUDE-RATE] wx, wy, wz", (sensors.gyro.x, sensors.gyro.y, sensors.gyro.z)) # Attitude rate, from gyroscope
    # 
    # # ########### ------------------ PRINT STATE -------------------- ########### 
    # 
    # # ########### ------------------ PRINT SETPOINT -------------------- ########### 
    # 
    # print("[SETPOINT][POSITION] x, y, z", (setpoint.position.x, setpoint.position.y, setpoint.position.z))
    # print("[SETPOINT][VELOCITY] vx, vy, vz", (setpoint.velocity.x, setpoint.velocity.y, setpoint.velocity.z))
    # print("[SETPOINT][ATTITUDE] r, p, y", (setpoint.attitude.roll, setpoint.attitude.pitch, setpoint.attitude.yaw))
    # print("[SETPOINT][ATTITUDE-RATE] wx, wy, wz", (setpoint.attitudeRate.roll, setpoint.attitudeRate.pitch, setpoint.attitudeRate.yaw))
    # 
    # # ########### ------------------ PRINT SETPOINT -------------------- ########### 

    ########### ------------------ SAVING THINGS -------------------- ###########

    data['STATE'] = {'POSITION': (state.position.x, state.position.y, state.position.z), 
                     'VELOCITY': (state.velocity.x, state.velocity.y, state.velocity.z),
                     'ATTITUDE': (state.attitude.roll, state.attitude.pitch, state.attitude.yaw)}
    data['SENSOR'] = {'ATTITUDE-RATE': (sensors.gyro.x, sensors.gyro.y, sensors.gyro.z)}
    data['SETPOINT'] = {'POSITION': (setpoint.position.x, setpoint.position.y, setpoint.position.z),
                        'VELOCITY': (setpoint.velocity.x, setpoint.velocity.y, setpoint.velocity.z),
                        'ATTITUDE': (setpoint.attitude.roll, setpoint.attitude.pitch, setpoint.attitude.yaw),
                        'ATTITUDE-RATE': (setpoint.attitudeRate.roll, setpoint.attitudeRate.pitch, setpoint.attitudeRate.yaw)}
    
    # Save data
    dataset[step] = data
    
    ########### ------------------ SAVING THINGS -------------------- ###########

    
    step += 1
    
import pickle, os

# Set to True if you want to collect data
collect_data = False

parent_folder = '../../datasets/EXP-3-CRAZYFLIE-CONTROLLERS-TEST'
folder = parent_folder + '/position'

if not os.path.isdir(folder):
    os.makedirs(folder)

########### ------------------ SAVING THINGS -------------------- ###########
if collect_data:
    print("Saving data...")
    with open(folder + '/data.pickle', 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Data saved in {folder}.")
########### ------------------ SAVING THINGS -------------------- ###########
    
