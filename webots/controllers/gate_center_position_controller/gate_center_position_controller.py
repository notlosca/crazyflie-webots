"""gate_corner_position_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot

import datetime, json

dataset_path = '../../datasets/EXP-1-NOPHYSICS-SPHERE'
collect_data = False
# create the Robot instance.
robot = Robot()
# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)

gps = robot.getDevice("gps")
gps.enable(timestep)
imu = robot.getDevice("inertial unit")
imu.enable(timestep)

sensor_name = robot.getName().split('_')[-1].upper()

saved = False
        
# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()
    print(imu.getRollPitchYaw())
    if not saved:
        sample = {sensor_name: {'xyz': gps.getValues()}}
        # Since the gate won't move, we can directly save its 
        # corner positions
        if collect_data:
            with open(f"{dataset_path}/gate_{sensor_name}.json", "w") as f:
                json.dump(sample, f)
        saved = True
        
    
    x, y, z = gps.getValues()
    # if len(sensor_name) > 2:
    #     print(f"[GATE {sensor_name}]\tX:{x:.4f}\tY:{y:.4f}\tZ:{z:.4f}")
    # else:
    #     print(f"[GATE {sensor_name} CORNER]\tX:{x:.4f}\tY:{y:.4f}\tZ:{z:.4f}")
    
    # Process sensor data here.
    
    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    pass

# Enter here exit cleanup code.
