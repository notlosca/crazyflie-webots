from controller import Supervisor
from controller import Motor
from controller import InertialUnit
from controller import GPS
from controller import Keyboard
from controller import Camera
from controller import DistanceSensor


from controller import CameraRecognitionObject

import os
from datetime import datetime
import cv2
import numpy as np
import random
import errno
import itertools
import transforms3d
import machinevisiontoolbox as mvtb
from spatialmath import SE3
import distinctipy
import scipy
import pickle

from math import cos, sin, degrees, radians
from ai import cs

import sys, shutil
sys.path.append('../../../controllers/')

# Import the path for the corner detection module
sys.path.append('../../../../scarciglia-nanodrone-gate-detection/')

from src import corner, rotation, geometry
from src import filter as flt

# fix seed to get always same color mapping
np.random.seed(0)

########### ------------------ SETTINGS ------------------ ###########

rotate_gate = True
if rotate_gate:
    rotate_gate_every = 100

change_gate_color = True
if change_gate_color:
    change_gate_color_every = 100

# change_gate_height = False
# if change_gate_height:
#     change_gate_height_every = 100
# 
# change_scale_gate = False
# if change_scale_gate:
#     change_scale_gate_every = 100

random_background = True
if random_background:
    random_background_every = 100

# This can be merged with random_background
random_floor = True
if random_floor:
    random_floor_every = 100

random_images = True
# img_path = '~/home/losca/Documents/Thesis/crazyflie-webots/webots/worlds/textures/textured_panel_3m/' # Not working
img_path = '../../worlds/textures/textured_panel_3m/'
available_imgs = os.listdir(img_path)
imgs = [i for i in available_imgs if i!='.DS_Store']
if random_images:
    random_images_every = 100

                ##### ------ LIMITS ------ #####

# Horizontal FOV of the camera. Used to limit the random yaw of the drone
horizontal_fov = 87 # deg

# Drone limits used to spawn it randomly
roll_limits = (np.deg2rad(-20), np.deg2rad(+20))
pitch_limits = (np.deg2rad(-10), np.deg2rad(+10))
yaw_limits = (np.deg2rad(-horizontal_fov/2), np.deg2rad(horizontal_fov/2))

drone_radius_limits = (0.5, 2.5)
z_limits = (0.05, 1.5)

                ##### ------ LIMITS ------ #####

########### ------------------ SETTINGS ------------------ ###########

########### ------------------ SAVING THINGS -------------------- ###########

# Set to True if you want to collect data
collect_data = True

# we store all the parameters and relevant experiment info in a dict
exp_dict = {"objects":{}, "env_objects":{}, "settings":{}}

# total number of samples
n_samples = 500
exp_dict['settings']['n_samples'] = n_samples

# name of the experiment
exp_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
if random_background:
    exp_name += "_around_random_bg"
else:
    exp_name += "_around_empty_bg"

exp_folder = f"../../datasets/simulation_datasets/{exp_name}"
imgs_folder = f"{exp_folder}/imgs/"
# gt_imgs_folder = f'{imgs_folder}' + 'ground_truth'
if collect_data:

    try:
        os.makedirs(imgs_folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..
        else:
            print(f"Overwriting folder {imgs_folder}")

exp_dict["settings"] = {
    "name": exp_name,
    "drone_spawn_limits":{
        "z_limits": z_limits,
        "yaw_limits": yaw_limits,
        "pitch_limits": pitch_limits,
        "roll_limits": roll_limits,
        "radius_limits": drone_radius_limits
        }
}

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
    camera.recognitionEnable(timestep)
    camera.enableRecognitionSegmentation()
    range_finder_full = robot.getDevice("range-finder-full")
    range_finder_full.enable(timestep)
    range_front = robot.getDevice("range_front")
    range_front.enable(timestep)
    range_left = robot.getDevice("range_left")
    range_left.enable(timestep)
    range_back = robot.getDevice("range_back")
    range_back.enable(timestep)
    range_right = robot.getDevice("range_right")
    range_right.enable(timestep)

    ########### ------------------ WEBOTS NODES ------------------ ###########

    # Crazyflie
    crazyflie_node = robot.getFromDef("CRAZYFLIE")
    translation_drone = crazyflie_node.getField('translation')
    rotation_drone = crazyflie_node.getField('rotation')
    camera_node = crazyflie_node.getField('children').getMFNode(2) # Be careful of the index !
    camera_drone_tr = camera_node.getField('translation').getSFVec3f()

                    ##### ------ CAMERA SETTINGS ------ #####

    f = camera.getFocalLength()
    pixel_size = (3.6e-6, 3.6e-6)
    fov = camera.getFov()
    img_size = (camera.getWidth(), camera.getHeight())
    cam = mvtb.CentralCamera(rho=pixel_size[0], imagesize=img_size, f=f)

                    ##### ------ CAMERA SETTINGS ------ #####

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

    ########### ------------------ WEBOTS NODES ------------------ ###########

    ########### ------------------ OBJECTS ------------------ ###########

    objects_node = robot.getFromDef("Objects")
    obj_colors = distinctipy.get_colors(objects_node.getField('children').getCount()) 

    obstacles = []
    curr_obstacle = 0
    walls_name = ["wall1", "wall2", "wall3", "wall4", "wall5", "wall6", "wall7"]

    floor_panel = {}

    for i, color in enumerate(obj_colors):
        solid = objects_node.getField('children').getMFNode(i)
        obj = {"color": solid.getField('recognitionColors').getMFColor(0)}  
        # obj['solid'] = solid  
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
        elif "Floor - Panel" in obj['name']:
            shape = solid.getField('children').getMFNode(0).getField('children').getMFNode(1).getField('children').getMFNode(0)
            pts_node = shape.getField('geometry').getSFNode().getField("coord").getSFNode().getField("point")
            pts = []
            for j in range(pts_node.getCount()):
                pts.append(pts_node.getMFVec3f(j))
            obj['points'] = pts
            floor_panel['starting_rotation'] = solid.getField('rotation').getSFRotation()
            floor_panel['solid'] = solid
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

        exp_dict["objects"][solid.getId()] = obj

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
    
    # # OpenCV show images
    # cv2.startWindowThread()
    # cv2.namedWindow("Drone Camera")

    colors = ['r', 'b', 'g', 'y']
    
    ########### ------------------ SAVING THINGS ------------------ ###########
    
    exp_dict['settings']['drone'] = {'starting_position': crazyflie_node.getField('translation').getSFVec3f(),
                        'starting_rotation': crazyflie_node.getField('rotation').getSFRotation(),
                        'camera_drone_tr': camera_drone_tr}
    
    exp_dict['settings']['gate'] = {'starting_corners':{'tl':tl, 'bl':bl, 'br':br, 'tr':tr},
                                    'starting_position': gate_node.getField('translation').getSFVec3f(),
                                    'starting_rotation': gate_node.getField('rotation').getSFRotation()}
    
    exp_dict['settings']['camera'] = {'f':f, 'pixel_size':pixel_size, 'img_size':img_size, 'fov':fov }
    
    # print(exp_dict)

    if collect_data:
        with open(f"{exp_folder}/exp_setup.pkl", "wb") as f:
            pickle.dump(exp_dict, f)

    ########### ------------------ SAVING THINGS ------------------ ###########
    
    it_idx = 0 # Iteration index

    frame_n = 0
    counter = -1
    samples = []

    # Main loop:
    while robot.step(timestep) != -1 and frame_n < n_samples:

        ## Get sensor data about the drone
        roll, pitch, yaw = imu.getRollPitchYaw()
        x_global, y_global, z_global = gps.getValues()

        # Dictionary containg data of the current sample
        sample = {}

        ########### ------------------ FLAGS ------------------ ###########

        if rotate_gate:
            if frame_n % rotate_gate_every == 0 and frame_n != 0:
                
                # Set the new orientation of the gate
                gate_yaw = np.random.uniform(0, 2*np.pi)
                rot = transforms3d.euler.euler2axangle(*(0,0, gate_yaw))
                ax_angle = list(rot[0])
                ax_angle.append(rot[-1]) 
                gate_node.getField('rotation').setSFRotation(ax_angle)
                
        if change_gate_color:
            if frame_n % change_gate_color_every == 0 and frame_n != 0:
                # Change RGB color
                r_rand = np.random.uniform(0,1, size=1)[0]
                g_rand = np.random.uniform(0,1, size=1)[0]
                b_rand = np.random.uniform(0,1, size=1)[0]

                # print([r_rand, g_rand, b_rand])
                # Gate
                gate_node = robot.getFromDef("GATE").getField('children').getMFNode(0)
                # The Color field of the gate is in Gate -> Children -> Solid "Gate" (idx = 5) -> Children -> Shape -> Appearance -> baseColor
                gate_node.getField('children').getMFNode(5).getField('appearance').getSFNode().getField('baseColor').setSFColor([r_rand, g_rand, b_rand])

        if random_floor:
            if frame_n % random_floor_every == 0 and frame_n != 0:

                original_axangle = floor_panel['starting_rotation']
                current_rot_mat = transforms3d.axangles.axangle2mat(original_axangle[:-1], angle=original_axangle[-1])

                # Perform a rotation around y axis -> a pitch rotation
                floor_pitch_limits = (-np.deg2rad(45), np.deg2rad(45))
                random_pitch = np.random.uniform(*floor_pitch_limits)
                new_rot_mat = current_rot_mat@rotation.rotation_matrix(roll=0,pitch=random_pitch,yaw=0)
                new_axangle = transforms3d.axangles.mat2axangle(new_rot_mat)
                ax_angle = list(new_axangle[0])
                ax_angle.append(new_axangle[-1]) 
                floor_panel['solid'].getField('rotation').setSFRotation(ax_angle)

        if random_background:
            if frame_n % random_background_every == 0 and frame_n != 0:

                # Clean the space radius of the drone
                # TODO: right now all the items near the gate are not there
                
                # Pick the panels and set them randomly
                for obs in obstacles:
                    if 'Panel - Textured' in obs[0].getField('name').getSFString():
                        # Compute the random spawn pose (translation and orientation)
                        # Gate
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
        
        if random_images:
            if frame_n % random_images_every == 0 and frame_n != 0:
                for obs in obstacles:
                    if 'Panel - Textured' in obs[0].getField('name').getSFString():
                        solid = obs[0]
                        shape1 = solid.getField('children').getMFNode(0).getField('children').getMFNode(1).getField('children').getMFNode(1) # I need the second shape, idx = 1
                        random_img_id = int(np.random.uniform(low=0, high=len(imgs)))
                        # Set the new image
                        shape1.getField('appearance').getSFNode().getField('baseColorMap').getSFNode().getField('url').setMFString(0, f'textures/textured_panel_3m/{imgs[random_img_id]}')
                    elif 'Floor - Panel' in obs[0].getField('name').getSFString():
                        solid = obs[0]
                        shape1 = solid.getField('children').getMFNode(0).getField('children').getMFNode(1).getField('children').getMFNode(1) # I need the second shape, idx = 1
                        random_img_id = int(np.random.uniform(low=0, high=len(imgs)))
                        # Set the new image
                        shape1.getField('appearance').getSFNode().getField('baseColorMap').getSFNode().getField('url').setMFString(0, f'textures/textured_panel_3m/{imgs[random_img_id]}')
                    
        # if change_gate_height:
        #     if frame_n % change_gate_height_every == 0 and frame_n != 0:
        #         height_lim = (-0.5, 0.5)
        #         random_height = np.random.uniform(*height_lim)
        #         gate_node = robot.getFromDef("GATE").getField('children').getMFNode(0)
        #         translation_gate = gate_node.getField('translation').getSFVec3f()
        #         gate_node.getField('translation').setSFVec3f([translation_gate[0], translation_gate[1], random_height])
        #         
        # if change_scale_gate:
        #     if frame_n % change_scale_gate_every == 0 and frame_n != 0:
        #         gate_scale_limits = (0.5, 2)
        #         random_scale = np.random.uniform(*gate_scale_limits)
        #         gate_node = robot.getFromDef("GATE").getField('children').getMFNode(0)
        #         gate_node.getField('children').getMFNode(0).getField('scale').setSFVec3f(3*[random_scale])
        
        ########### ------------------ FLAGS ------------------ ###########


        ########### ------------------ CORNERS - GROUND TRUTH ------------------ ###########

        # Gate
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

        # It raises problems
        # T_C = SE3(wd_tr)*SE3.RPY(roll,pitch,yaw)*SE3(dc_tr)*SE3.RPY(-np.pi/2, 0, -np.pi/2)
        # With mine it works
        T_C = extrinsic_matrix
        # Ground truth projected into the image space
        GT_p_detected = cam.project_point(P, pose=SE3(T_C, check=False))
        # print(GT_p_detected)
        
        ########### ------------------ CORNERS - GROUND TRUTH ------------------ ###########
        
        
        ########### ------------------ CAMERA IMAGES -------------------- ########### 

        w, h = camera.getWidth(), camera.getHeight()
        cameraData = camera.getImage()  # Note: uint8 string
        segData = camera.getRecognitionSegmentationImage() # Segmentation image

        image = np.frombuffer(cameraData, np.uint8).reshape(h, w, 4) # BGR, alpha (transparency)
        seg = np.frombuffer(segData, np.uint8).reshape(h, w, 4)

        cam_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # gray-scale image
        
        w, h = range_finder_full.getWidth(), range_finder_full.getHeight()
        rangeData = range_finder_full.getRangeImage(data_type="list")
        range_image_full = np.array(rangeData, np.float32).reshape(h, w)

        ########### ------------------ CAMERA IMAGES -------------------- ########### 


        ########### ------------------ SAVING THINGS -------------------- ###########

        # Check whether if the ground truths are present in the image. If yes save everything
        if np.isnan(GT_p_detected).sum() == 0 and np.min(GT_p_detected) >= 0 and np.max(GT_p_detected) < img_size[0]:
            # gt_img = np.zeros_like(image)
            # # Show image
            # for id, col in enumerate(colors):
            #     v, u = GT_p_detected[:,id] # Ground truth into image space
            #     image = cv2.circle(image, (int(v),int(u)), radius=int(np.round(0.05*320,0)), color=(255, 255, 255), thickness=-1)
            #     # gt_img = cv2.circle(gt_img, (int(v),int(u)), radius=int(np.round(0.05*320,0)), color=(255, 255, 255), thickness=-1)
            #     gt_img = cv2.circle(gt_img, (int(v),int(u)), radius=10, color=(255, 255, 255), thickness=-1)
            #     # gt_img = cv2.blur(gt_img, (7,7))
            #     gt_img = scipy.ndimage.gaussian_filter(gt_img, sigma=1)
            #     # image = cv2.putText(image, text=str(id), org = (int(x),int(y)), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.5, color = (255,255,255), thickness = 1)
            
            # # Resize for the ground truth
            # image = cv2.resize(gt_img, (20,20))
        
            if collect_data:
                # # Save the image
                # cv2.imwrite(f"{gt_imgs_folder}"+f'/img_{frame_n}.png', cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
                
                # To be coherent with the drone orientation, transform the axangle orientation into roll, pitch and yaw
                gate_axangle = gate_node.getField('rotation').getSFRotation()[:-1], gate_node.getField('rotation').getSFRotation()[-1]
                gate_rpy = transforms3d.euler.axangle2euler(gate_axangle[0], gate_axangle[-1])

                # Save the data
                sample["sample_n"] = frame_n
                sample['image'] = cam_img
                sample['gate'] = {'position':gate_node.getField('translation').getSFVec3f(), 
                                'orientation':list(gate_rpy)}
                sample["drone"] = {'position':gps.getValues(),
                                'orientation': imu.getRollPitchYaw(),}
                sample['GT_3D_space'] = P
                sample['GT_image_space'] = GT_p_detected
                
                # rec_objects = []
                # 
                # # Store info about objects recognized by the camera
                # for rec_obj in camera.getRecognitionObjects():
                #     # https://www.cyberbotics.com/doc/reference/camera?tab-language=python#camera-recognition-object
                #     obj = {	
                #     "w_id": rec_obj.getId(),
                #     "camera_p": rec_obj.getPosition(),
                #     "camera_o": rec_obj.getOrientation(),
                #     "camera_size": rec_obj.getSize(),
                #     "image_p": rec_obj.getPositionOnImage(),
                #     "image_size": rec_obj.getSizeOnImage(),
                #     "name": robot.getFromId(rec_obj.getId()).getField('name').getSFString()
                #     }
                #     rec_objects.append(obj)
                # sample["rec_objects"] = rec_objects
                samples.append(sample)

                path = f"{imgs_folder}/img_{frame_n}.png"
                cv2.imwrite(path, cam_img)
                path = f"{imgs_folder}/seg_{frame_n}.png"
                cv2.imwrite(path, seg)
                
                # r_i = []
                # for els in range_image:
                #     r_i_row = []
                #     for el in els:
                #         if el == float('inf'):
                #             el = 4.0
                #         
                #         r_i_row.append(int(el*255/4.0))
                #     r_i.append(r_i_row)
                # cv2.imwrite(path, np.asarray(r_i, dtype=np.uint8))

                path = f"{imgs_folder}" + f"range_full_{frame_n}.png"
                np.save(path, range_image_full)
            
            # Increase the counter of the correct frames
            frame_n += 1
        
        ########### ------------------ SAVING THINGS -------------------- ########### 


        ########### ------------------ NEW DRONE SPAWN POINT ------------------ ###########

        gate_pos = np.array(gate_center)
        gate_orientation = gate_rot
        
        # Spherical coordinates angle limits
        theta_lim = (0, np.pi)
        phi_lim = (0, 2*np.pi)
        
        origin = np.array(gate_pos)
        x0,y0,z0 = origin

        r = np.random.uniform(*drone_radius_limits)
        theta = np.random.uniform(theta_lim[0], theta_lim[1])
        phi = np.random.uniform(phi_lim[0], phi_lim[1])

        # Consider x and y as a cilinder. I don't want that the drone can bu underneath the gate centre
        x = x0 + r*np.cos(phi)
        y = y0 + r*np.sin(phi)
        z = z0 + r*np.cos(theta)

        z = np.clip(z, z_limits[0], z_limits[1])
        
        point = np.array([x,y,z])

        delta_array = gate_pos - point
        r, pitch, yaw = cs.cart2sp(x=delta_array[0], y=delta_array[1], z=delta_array[-1])

        pitch = np.clip(pitch, *pitch_limits)
        
        rot = transforms3d.euler.euler2axangle(*(0, -pitch, yaw))
        ax_angle = list(rot[0])
        ax_angle.append(rot[-1])      

        translation_drone.setSFVec3f(list(point))

        crazyflie_node.getField('rotation').setSFRotation(ax_angle)
        crazyflie_node.resetPhysics()

        ########### ------------------ NEW DRONE SPAWN POINT ------------------ ###########
        
        it_idx += 1

        print(frame_n, it_idx)
        # print(f'Percentage of correct images: {100*(frame_n/it_idx):.2f}')
    

    ########### ------------------ SAVING THINGS -------------------- ###########

    if collect_data:
        with open(f"{exp_folder}/samples.pkl", "wb") as f:
            pickle.dump(samples, f)
            print(f"Data saved in {exp_folder}.")

    ########### ------------------ SAVING THINGS -------------------- ###########
            