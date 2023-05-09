"""my_supervisor controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Supervisor
import numpy as np
import json
import itertools
import transforms3d
from ai import cs

robot = Supervisor()  # create Supervisor instance

# Set to True if you want to collect data
collect_data = False

dataset_path = '../../datasets/EXP-1-NOPHYSICS-SPHERE'

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
gate_center = gate_node.getField('children').getMFNode(4).getField('translation')

# Save data

if collect_data:

    with open(f"{dataset_path}/drone_settings.json", "w") as f:
        json.dump(settings , f)

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

i = 0
while robot.step(timestep) != -1 and i < len(clean_array):
    gate_pos = translation_gate.getSFVec3f()
    center = np.array(gate_pos) + np.array(gate_center.getSFVec3f())
    x0, y0, z0 = center
    
    delta_array = center - clean_array[i] # vector pointing to the gate center (the origin)
    # theta = pitch, phi = yaw
    r, pitch, yaw = cs.cart2sp(x=delta_array[0], y=delta_array[1], z=delta_array[-1])
    roll = 0
    
    drone_coord = clean_array[i] + np.array(gate_pos)
    
    rot = transforms3d.euler.euler2axangle(roll, -pitch, yaw)
    ax_angle = list(rot[0])
    ax_angle.append(rot[-1])      

    translation_drone.setSFVec3f(list(drone_coord))

    crazyflie_node.getField('rotation').setSFRotation(ax_angle)
    #print(crazyflie_node.getField('rotation').getSFRotation())
    crazyflie_node.resetPhysics()
    print(f'{i}/{len(clean_array)}')
    i += 1