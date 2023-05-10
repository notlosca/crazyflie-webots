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

# Desired states
tasks = {}

take_off = True
take_off_info = {'setpoints': {'velocity.x':0.0, 'velocity.y':0.0, 'position.z':1.0, 'attitudeRate.yaw':0.0}}
tasks[0] = take_off_info

first_task = False
first_task_info = {'setpoints': {'position.x':1.0, 'position.y':1.0, 'position.z':1.0, 'attitudeRate.yaw':0}, 'num_steps':1000}
first_task_step = 0
tasks[1] = first_task_info

second_task = False

prev_step = False
hovering_steps = 0 # Counter used to check how many times we are in the desired state.

dataset = {'info':tasks}

# dataset = {'info':{'setpoint':'position', 'x':x_desired, 'y':y_desired, 'z':z_desired, 'yaw':'disabled'}}
# dataset = {'info':{'setpoint':'velocity', 'vx':0.1, 'vy':0.1, 'vz':0.1}}
# dataset = {'info':{'setpoint':'attitude_rate (rad/s)', 'roll':8, 'pitch':8, 'yaw':8, 'z':1.0}}

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


# Main loop:
while robot.step(timestep) != -1: # and step < 5000:
    # print(step)
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

    if take_off:
                
        info = tasks[0] # take_off_info
        
        isclose = np.isclose(zGlobal, info['setpoints']['position.z'], rtol=1e-2)
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
        
        ## Fill in Setpoints
        # Setpoint mode
        setpoint = cffirmware.setpoint_t()
        
        setpoint.mode.x = cffirmware.modeVelocity
        setpoint.mode.y = cffirmware.modeVelocity
        
        setpoint.mode.z = cffirmware.modeAbs
        
        setpoint.mode.yaw = cffirmware.modeVelocity
        
        setpoint.position.z = info['setpoints']['position.z']
        
        setpoint.velocity.x = info['setpoints']['velocity.x']
        setpoint.velocity.y = info['setpoints']['velocity.y']
        
        my_vel = info['setpoints']['attitudeRate.yaw']
        # magic_constant = 0.0626
        magic_constant = 1
        setpoint.attitudeRate.yaw = degrees(my_vel/magic_constant)
        
        setpoint.velocity_body = True
    
        print("Desired state", info)
    
    elif first_task:
                
        info = tasks[1]
        
        ## Fill in Setpoints
        # Setpoint mode
        setpoint = cffirmware.setpoint_t()
        
        setpoint.mode.x = cffirmware.modeAbs
        setpoint.mode.y = cffirmware.modeAbs
        setpoint.mode.z = cffirmware.modeAbs
        
        setpoint.mode.yaw = cffirmware.modeVelocity
        
        setpoint.position.x = info['setpoints']['position.x']
        setpoint.position.y = info['setpoints']['position.y']
        setpoint.position.z = info['setpoints']['position.z']
        
        
        my_vel = info['setpoints']['attitudeRate.yaw']
        # magic_constant = 0.0626 # = 1/16 ca.
        magic_constant = 1
        setpoint.attitudeRate.yaw = degrees(my_vel/magic_constant)
        
        setpoint.velocity_body = True
    
        print("Desired state", info)

        print("########### ------------------ GPS [m] -------------------- ###########")
        print(f"X: {xGlobal:.4f}\tY: {yGlobal:.4f}\tZ: {zGlobal:.4f}")
        print("########### ------------------ GPS [m] -------------------- ###########")
        print("\n ")
        
        first_task_step += 1
        
        if first_task_step == info['num_steps']:
            first_task = True
            print("Passing to the next task...")
            break
        
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
collect_data = True

parent_folder = '../../datasets/EXP-3-CRAZYFLIE-CONTROLLERS-TEST'
folder = parent_folder +'/tests'+ '/04_firmware_test'

if not os.path.isdir(folder):
    os.makedirs(folder)

########### ------------------ SAVING THINGS -------------------- ###########
if collect_data:
    print("Saving data...")
    with open(folder + '/data.pickle', 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Data saved in {folder}.")
########### ------------------ SAVING THINGS -------------------- ###########
    

