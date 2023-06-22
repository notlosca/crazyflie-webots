from controller import Supervisor
from controller import Motor
from controller import InertialUnit
from controller import GPS
from controller import Keyboard
from controller import Camera
from controller import DistanceSensor

import os, datetime
import cv2
import numpy as np
import random
import errno
import itertools
import transforms3d
import machinevisiontoolbox as mvtb
from spatialmath import SE3
import distinctipy


from math import cos, sin, degrees, radians
from ai import cs

import sys, shutil
sys.path.append('../../../controllers/')

# Import the path for the corner detection module
sys.path.append('../../../../scarciglia-nanodrone-gate-detection/')

from src import corner, rotation, geometry
from src import filter as flt

np.random.seed(0)

########### ------------------ SETTINGS ------------------ ###########

rotate = False
if rotate:
    rotate_gate_every = 100

change_gate_color = False
if change_gate_color:
    change_gate_color_every = 100

change_gate_height = False
if change_gate_height:
    change_gate_height_every = 100

change_scale_gate = False
if change_scale_gate:
    change_scale_gate_every = 100

random_background = False
if random_background:
    random_background_every = 100

########### ------------------ SETTINGS ------------------ ###########


########### ------------------ SAVING THINGS -------------------- ###########

# Set to True if you want to collect data
collect_data = False

if collect_data:
        
    parent_folder = '../../datasets/EXP-6-IBVS_SMOOTH_START'
    folder = parent_folder +'/tests_elia/'+ '04_corner_det_test_no_filter_allsmooth_1percsmooth'

    imgs_folder = f'{folder}/imgs/'

    try:
        if os.path.isdir(imgs_folder):
            shutil.rmtree(imgs_folder)
        os.makedirs(imgs_folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..
        else:
            print(f"Overwriting folder {imgs_folder}")
    
########### ------------------ SAVING THINGS -------------------- ###########

if __name__ == '__main__':

    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    
    sampling_frequency = timestep
    
    ## Initialize Sensors
    imu = robot.getDevice("inertial unit")
    imu.enable(timestep)
    gps = robot.getDevice("gps")
    gps.enable(timestep)
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
    gate_node = robot.getFromDef("GATE").getField('children').getMFNode(0)
    translation_gate = gate_node.getField('translation').getSFVec3f()
    # Gate_center in GATE -> children -> Solid 'Gate' -> children
    gate_center = gate_node.getField('children').getMFNode(0).getField('translation').getSFVec3f()
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

    dataset = {'info':None, 'sampling_frequency':sampling_frequency, 'corner_detection': False}

    ########### ------------------ OBJECTS ------------------ ###########

    objects_node = robot.getFromDef("Objects")
    obj_colors = distinctipy.get_colors(objects_node.getField('children').getCount()) 

    obstacles = []
    curr_obstacle = 0
    walls_name = ["wall1", "wall2", "wall3", "wall4", "wall5", "wall6", "wall7"]
    for i, color in enumerate(obj_colors):
        solid = objects_node.getField('children').getMFNode(i)
        obj = {"color": solid.getField('recognitionColors').getMFColor(0)}    
        obj['name'] = solid.getField('name').getSFString()
        
        obj['translation'] = solid.getField('translation').getSFVec3f()
        obj['rotation'] = solid.getField('rotation').getSFRotation()
        obj['scale'] = solid.getField('scale').getSFVec3f()

        if "Panel - Textured" in obj["name"] or "Mat - " in obj["name"] or "Pole - " in obj["name"]:
            # bbox info are taken from child -> $CollaAutoNames$_0 (index 0)-> child -> ID3 (index 1) -> child -> Shape at index 0?
            shape = solid.getField('children').getMFNode(0).getField('children').getMFNode(1).getField('children').getMFNode(0)
            pts_node = shape.getField('geometry').getSFNode().getField("coord").getSFNode().getField("point")
            pts = []
            for j in range(pts_node.getCount()):
                pts.append(pts_node.getMFVec3f(j))
            obj['points'] = pts
        elif "Panel - Metal" in obj["name"] or "Gate" in obj["name"] or "Curtain - Striped" in obj["name"]:
            # bbox is taken from all shapes in child -> $CollaAutoNames$_0 (index 0)-> child -> ID3 (index 1) -> child -> shape group -> children
            shapes = solid.getField('children').getMFNode(0).getField('children').getMFNode(1).getField('children').getMFNode(0).getField('children')
            pts = []
            for n in range(shapes.getCount()):
                pts_node = shapes.getMFNode(n).getField('geometry').getSFNode().getField("coord").getSFNode().getField("point")
                for j in range(pts_node.getCount()):
                    pts.append(pts_node.getMFVec3f(j))
            obj['points'] = pts
        elif "person" in obj["name"]:
            shape = solid.getField('children').getMFNode(0)
            pts_node = shape.getField('geometry').getSFNode().getField("coord").getSFNode().getField("point")
            pts = []
            for j in range(pts_node.getCount()):
                pts.append(pts_node.getMFVec3f(j))
            obj['points'] = pts
        elif "solid" in obj["name"]:
            shape = solid.getField('children').getMFNode(0)
            pts_node = shape.getField('geometry').getSFNode().getField("coord").getSFNode().getField("point")
            pts = []
            for j in range(pts_node.getCount()):
                pts.append(pts_node.getMFVec3f(j))
            obj['points'] = pts

        # # this part is commented out because it can be run just once to set the objects segmentation color
        # # then the scene can be saved
        # # set recognition color
        # while solid.getField('recognitionColors').getCount() != 0:
        #     solid.getField('recognitionColors').removeMF(0)
        # solid.getField('recognitionColors').insertMFColor(0, list(color))
        # solid_string = solid.exportString()
        # objects_node.getField('children').removeMF(i)
        # objects_node.getField('children').importMFNodeFromString(i, solid_string)

        # we ignore the poster on the wall (we can't put it in the middle of the arena)
        if "Curtain - Striped" not in obj['name']:
            obstacles.append((solid, pts, obj['scale']))
    
    ########### ------------------ OBJECTS ------------------ ###########
    
    # Horizontal FOV of the camera. Used to limit the random yaw of the drone
    horizontal_fov = 87 # deg
    
    # Drone limits used to spawn it randomly
    roll_limits = (np.deg2rad(-20), np.deg2rad(+20))
    pitch_limits = (np.deg2rad(-20), np.deg2rad(+20))
    yaw_limits = (np.deg2rad(-horizontal_fov/2), np.deg2rad(horizontal_fov/2))

    radius_limits = (0.5, 2.5)
    z_limits = (0.05, 2)

    ########### ------------------ VISUAL SERVOING ------------------ ###########
    
    f = 0.0006
    pixel_size = (3.6e-6, 3.6e-6)
    img_size = (320,320)
    cam = mvtb.CentralCamera(rho=pixel_size[0], imagesize=img_size, f=f)

    # Depth value
    Z = 0.34

    lmda = 0.08

    thresh = 5e-1
    thresh = 8
    err = np.inf

    ########### ------------------ VISUAL SERVOING ------------------ ###########
    
    # OpenCV show images
    cv2.startWindowThread()
    cv2.namedWindow("Drone Camera")

    colors = ['r', 'b', 'g', 'y']
    
    ########### ------------------ SAVING THINGS ------------------ ###########
    
    dataset['drone'] = {'starting_position': crazyflie_node.getField('translation').getSFVec3f(),
                        'starting_rotation': crazyflie_node.getField('rotation').getSFRotation(),
                        'camera_drone_tr': camera_node.getField('translation').getSFVec3f()}
    dataset['gate'] = {'corners':{'tl':tl, 'bl':bl, 'br':br, 'tr':tr},
                    'position': gate_node.getField('translation').getSFVec3f(),
                        'rotation': gate_node.getField('rotation').getSFRotation()}
    dataset['camera'] = {'f':f, 'pixel_size':pixel_size, 'img_size':img_size }
    dataset['ibvs'] = {'lambda': lmda, 'threshold': thresh, 'Z':{'estimated':False, 'value':Z},}
    
    ########### ------------------ SAVING THINGS ------------------ ###########
    
    it_idx = 0 # Iteration index
        
    theta_lim = (0,np.pi/4)
    phi_lim = (0,2*np.pi)

    cnt_ok_pos = 0
        
    # Main loop:
    while robot.step(timestep) != -1:
        # if it_idx == 0:
        #     it_idx += 1
        #     continue
        data = {}
        
        ## Get sensor data
        roll, pitch, yaw = imu.getRollPitchYaw()
        altitude = gps.getValues()[2]
        x_global, y_global, z_global = gps.getValues()

        ########### ------------------ SAVING THINGS -------------------- ###########

        data['GPS'] = gps.getValues()
        data['IMU'] = imu.getRollPitchYaw()
        
        ########### ------------------ SAVING THINGS -------------------- ###########
        
        # CAMERA IMAGES
        w, h = camera.getWidth(), camera.getHeight()
        cameraData = camera.getImage()  # Note: uint8 string
        image = np.frombuffer(cameraData, np.uint8).reshape(h, w, 4) # BGR, alpha (transparency)
        
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # gray-scale image
    
        if collect_data:
            
            # Save image
            path = f'{imgs_folder}/img_{it_idx}.png'
            cv2.imwrite(path, img) # gray-scale image
    
        ########### ------------------ FLAGS ------------------ ###########

        if rotate:
            if cnt_ok_pos % rotate_gate_every == 0:
                # Rotate the gate
                gate_yaw = np.random.uniform(0, 2*np.pi)
                rot = transforms3d.euler.euler2axangle(*(0,0, gate_yaw))
                ax_angle = list(rot[0])
                ax_angle.append(rot[-1]) 
                gate_node.getField('rotation').setSFRotation(ax_angle)
                
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
                print(rot)
                print('Gate data (br, bl, tl, tr, gate_center, translation_gate):', br, bl, tl, tr, gate_center, translation_gate)
        
        if change_gate_color:
            if cnt_ok_pos % change_gate_color_every == 0:
                # Change RGB color
                r_rand = np.random.uniform(0,1, size=1)[0]
                g_rand = np.random.uniform(0,1, size=1)[0]
                b_rand = np.random.uniform(0,1, size=1)[0]

                print([r_rand, g_rand, b_rand])
                # Gate
                gate_node = robot.getFromDef("GATE")
                # The Color field of the gate is in Gate -> Children -> Solid "Gate" (idx = 5) -> Children -> Shape -> Appearance -> baseColor
                gate_node.getField('children').getMFNode(5).getField('children').getMFNode(0).getField('appearance').getSFNode().getField('baseColor').setSFColor([r_rand, g_rand, b_rand])

        if random_background:
            if cnt_ok_pos % random_background_every == 0:

                # Clean the space radius of the drone
                # TODO: right now all the items near the gate are not there
                
                # Pick the panels and set them randomly
                for obs in obstacles:
                    if 'Panel - Textured' in obs[0].getField('name').getSFString():
                        # Compute the random spawn pose (translation and orientation)
                        # Gate
                        gate_node = robot.getFromDef("GATE")
                        translation_gate = gate_node.getField('translation').getSFVec3f()
                        gate_center = gate_node.getField('children').getMFNode(4).getField('translation').getSFVec3f()
                        gate_rot = gate_node.getField('rotation').getSFRotation()
                        
                        # Position
                        radius_limits = (2.5, 3.5)
                        r = np.random.uniform(*radius_limits)
                        phi = np.random.uniform(0,2*np.pi)
                        
                        x = gate_center[0] + r*np.cos(phi) 
                        y = gate_center[1] + r*np.sin(phi)
                        z = -0.5

                        random_pt = [x,y,z]
                        obs[0].getField('translation').setSFVec3f(random_pt)
                        
                        # Orientation
                        delta_array = np.array(gate_center) - np.array(random_pt)
                        r, pitch, yaw = cs.cart2sp(x=delta_array[0], y=delta_array[1], z=delta_array[-1])

                        R = rotation.rotation_matrix(roll=0, pitch=0, yaw=yaw + np.pi/2) # To be able to show the image texture
                        axangle = transforms3d.euler.euler2axangle(*transforms3d.euler.mat2euler(R))
                        ax_angle = list(axangle[0])
                        ax_angle.append(axangle[-1]) 
                        obs[0].getField('rotation').setSFRotation(ax_angle)
                        
                        # Randomly scale the panels
                        scale_limits = (0.03, 0.1)
                        scale = np.random.uniform(*scale_limits)
                        obs[0].getField('scale').setSFVec3f(3*[scale])
                # Randomly change the image of the panels
                # pass

        if change_gate_height:
            if cnt_ok_pos % change_gate_height_every == 0:
                height_lim = (-0.5, 0.5)
                random_height = np.random.uniform(*height_lim)
                gate_node = robot.getFromDef("GATE")
                translation_gate = gate_node.getField('translation').getSFVec3f()
                gate_node.getField('translation').setSFVec3f([translation_gate[0], translation_gate[1], random_height])
                
        if change_scale_gate:
            if cnt_ok_pos % change_scale_gate_every == 0:
                gate_scale_limits = (0.5, 2)
                random_scale = np.random.uniform(*gate_scale_limits)
                gate_node = robot.getFromDef("GATE")
                gate_node.getField('children').getMFNode(0).getField('scale').setSFVec3f(3*[random_scale])
        ########### ------------------ FLAGS ------------------ ###########

        ########### ------------------ CORNERS - GROUND TRUTH ------------------ ###########
        # Gate
        gate_node = robot.getFromDef("GATE").getField('children').getMFNode(0)
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
        rotation_matrix_world_drone = rotation.rotation_matrix(roll, pitch, yaw)
        wd_tr = np.array([x_global, y_global, z_global])
        extrinsic_matrix_world_drone = rotation.get_extrinsic_matrix(rotation_matrix_world_drone, translation_vector=wd_tr)
        
        rotation_matrix_drone_camera = rotation.rotation_matrix(roll=-np.pi/2, pitch=0, yaw=-np.pi/2)
        dc_tr = camera_drone_tr
        extrinsic_matrix_drone_camera = rotation.get_extrinsic_matrix(rotation_matrix_drone_camera, translation_vector=dc_tr)

        extrinsic_matrix = extrinsic_matrix_world_drone@extrinsic_matrix_drone_camera

        T_C = SE3(wd_tr)*SE3.RPY(roll,pitch,yaw)*SE3(dc_tr)*SE3.RPY(-np.pi/2, 0, -np.pi/2)

        GT_p_detected = cam.project_point(P, pose=SE3(T_C, check=False)) 
        
        if np.isnan(GT_p_detected).sum() == 0 and np.min(GT_p_detected) >= 0 and np.max(GT_p_detected) <= 319:
            print(np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw))
            gt_img = np.zeros_like(image)
            # Show image
            for id, col in enumerate(colors):
                x, y = GT_p_detected[:,id] # Ground truth
                image = cv2.circle(image, (int(x),int(y)), radius=int(np.round(0.05*320,0)), color=(255, 255, 255), thickness=-1)
                gt_img = cv2.circle(gt_img, (int(x),int(y)), radius=int(np.round(0.05*320,0)), color=(255, 255, 255), thickness=-1)
                gt_img = cv2.blur(gt_img, (7,7))
                # image = cv2.putText(image, text=str(id), org = (int(x),int(y)), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.5, color = (255,255,255), thickness = 1)
                if collect_data:
                    # Save the image
                    cv2.imwrite(imgs_ibvs_folder+f'/img_{it_idx}.png', cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            
            # Resize for the ground truth
            image = cv2.resize(gt_img, (20,20))
            
            cv2.imshow("Drone Camera", cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            cv2.waitKey(timestep)
            cnt_ok_pos += 1
            print(cnt_ok_pos)

        ########### ------------------ CORNERS - GROUND TRUTH ------------------ ###########
        

        # Save data
        dataset[it_idx] = data

        gate_pos = np.array(gate_center)
        gate_orientation = gate_rot
        
        theta_lim = (0, np.pi)
        phi_lim = (0, 2*np.pi)
        
        radius_limits = (0.5, 2.0)
        origin = np.array(gate_pos)
        x0,y0,z0 = origin

        r = np.random.uniform(radius_limits[0], radius_limits[1])
        theta = np.random.uniform(theta_lim[0], theta_lim[1])
        phi = np.random.uniform(phi_lim[0], phi_lim[1])

        # x = x0 + r*np.sin(theta)*np.cos(phi)
        # y = y0 + r*np.sin(theta)*np.sin(phi)
        # z = z0 + r*np.cos(theta)

        # Consider x and y as a cilinder. I don't want that the drone can bu underneath the gate centre
        x = x0 + r*np.cos(phi)
        y = y0 + r*np.sin(phi)
        z = z0 + r*np.cos(theta)

        z = np.clip(z, z_limits[0], z_limits[1])
        
        point = np.array([x,y,z])

        delta_array = gate_pos - point
        r, pitch, yaw = cs.cart2sp(x=delta_array[0], y=delta_array[1], z=delta_array[-1])

        pitch = np.clip(pitch, *pitch_limits)
        yaw = np.random.uniform(yaw + yaw_limits[0], yaw + yaw_limits[1])
        # pitch = np.random.uniform(pitch_lim[0], pitch_lim[1])

        # rot = (np.random.uniform(roll_limits[0], roll_limits[1]), np.random.uniform(pitch_limits[0], pitch_limits[1]), np.random.uniform(yaw_to_obs + yaw_limits[0], yaw_to_obs + yaw_limits[1]))
        rot = transforms3d.euler.euler2axangle(*(0, -pitch, yaw))
        ax_angle = list(rot[0])
        ax_angle.append(rot[-1])      

        translation_drone.setSFVec3f(list(point))

        crazyflie_node.getField('rotation').setSFRotation(ax_angle)
        crazyflie_node.resetPhysics()
        it_idx += 1

        print(cnt_ok_pos, it_idx)
    import pickle, os

    ########### ------------------ SAVING THINGS -------------------- ###########

    if collect_data:
        
        print("Saving data...")
        with open(folder + '/data.pickle', 'wb') as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Data saved in {folder}.")
    ########### ------------------ SAVING THINGS -------------------- ###########
            