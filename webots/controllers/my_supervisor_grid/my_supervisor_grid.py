"""my_supervisor controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Supervisor
import numpy as np
import json
import itertools
import transforms3d

robot = Supervisor()  # create Supervisor instance

# Set to True if you want to collect data
collect_data = False

dataset_path = '../../datasets/EXP-1-NOPHYSICS-GRID'

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

TIME_STEP = 32

root_node = robot.getRoot()

# Store relevant informations
settings = {}


crazyflie_node = robot.getFromDef("CRAZYFLIE")
translation_drone = crazyflie_node.getField('translation')
rotation_drone = crazyflie_node.getField('rotation')
camera_drone_tr = crazyflie_node.getField('children').getMFNode(1).getField('translation').getSFVec3f()

settings['drone-camera-translation-vector'] = camera_drone_tr

gate_node = robot.getFromDef("GATE")
translation_gate = gate_node.getField('translation')

# range limitations of roll, pitch, and yaw to sample from for drone poses
roll_lim = (np.radians(-15), np.radians(15))
pitch_lim = (np.radians(-15), np.radians(15))
yaw_lim = (np.radians(-60 + 90), np.radians(60 + 90)) # offset to take into account the NWU coordinate of the drone while world frame is ENU

x_lim = (-3, +3)
y_lim = (-5, +0)
z_lim = (0, +2)

# create a grid around the gate
x_grid = np.linspace(x_lim[0], x_lim[1], num=2*(x_lim[1] - x_lim[0] + 1), endpoint=True)
y_grid = np.linspace(y_lim[0], y_lim[1], num=2*(y_lim[1] - y_lim[0] + 1), endpoint=True)
z_grid = np.linspace(z_lim[0], z_lim[1], num=2*(z_lim[1] - z_lim[0] + 1), endpoint=True)
grid_coord = list(itertools.product(x_grid,y_grid,z_grid))

# Save data

if collect_data:

    with open(f"{dataset_path}/drone_settings.json", "w") as f:
        json.dump(settings , f)

i = 0
while robot.step(timestep) != -1 and i < len(grid_coord):
    gate_pos = translation_gate.getSFVec3f()

    drone_coord = np.array(grid_coord[i]) + np.array(gate_pos)
    #print(drone_coord)
    #print(gate_pos)
    #drone_coord = np.zeros(shape=(3,))
    random_rot_dict = {'roll': np.random.uniform(*roll_lim), 'pitch': np.random.uniform(*pitch_lim), 'yaw': np.random.uniform(*yaw_lim)}
    rot = transforms3d.euler.euler2axangle(*random_rot_dict.values())
    ax_angle = list(rot[0])
    ax_angle.append(rot[-1])      

    translation_drone.setSFVec3f(list(drone_coord))

    crazyflie_node.getField('rotation').setSFRotation(ax_angle)
    #print(crazyflie_node.getField('rotation').getSFRotation())
    crazyflie_node.resetPhysics()
    
    i += 1
#robot.simulationQuit(1)
print(100*'----_____')
  #if i == 5:
  #    new_value = [-1,-1,0]
  #    translation_field.setSFVec3f(new_value)
  #if i == 200:
  #    crazy_position = crazyflie_node.getPosition()
  #    new_value = [crazy_position[0], crazy_position[1], 2.5]
  #    translation_field.setSFVec3f(new_value)
  #if i == 200:
  #    crazyflie_node.remove()
  #if i == 100:
  #    children_field.importMFNodeFromString(-1, 'CRAZYFLIE {translation 1 1 1 }')
   
