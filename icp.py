"""
pcd_obs_env with:
1. object/background segmentation
2. object registration
3. goal sampling
4. reward calculation
"""

import numpy as np
from PIL import Image as im 
import os
import argparse
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import open3d as o3d
import numpy as np
from ctypes import * # convert float to uint32
# from matplotlib import pyplot as plt
import copy
import torch

import rospy
import rosbag
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from numpy.linalg import inv
from lib_cloud_conversion_between_Open3D_and_ROS import convertCloudFromRosToOpen3d
from scipy.spatial.transform import Rotation
import copy
#import fpsample

# ICP
from KD_tree import *
from numpy.lib.function_base import diff
from registration_3d import *
from cartesian import *
colors = ['tab:red' ,'tab:blue', 'tab:orange', 'tab:green']

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])

def visualize_pcds(pcds):

    coor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    coor_frame.scale(0.2, center=(0.,0.,0.))
    vis.add_geometry(coor_frame)
    vis.get_render_option().background_color = np.asarray([255, 255, 255])

    view_ctl = vis.get_view_control()
    for pcd in pcds:
        vis.add_geometry(pcd)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()


    view_ctl.set_up((1, 0, 0))  # set the positive direction of the x-axis as the up direction
    view_ctl.set_front((-0.3, 0.0, 0.2))  # set the positive direction of the x-axis toward you
    view_ctl.set_lookat((0.0, 0.0, 0.3))  # set the original point as the center point of the window
    vis.run()
    vis.destroy_window()

def get_start_pose(object_pcd, standard_pcd, use_min_z = False , consider_shape = False, standard_file = None):

    xyz = np.array( object_pcd.points )
    x = np.mean(xyz[:,0])
    y = np.mean(xyz[:,1])
    z = np.mean(xyz[:,2])

    # print("pose: ", pose)
    if(use_min_z):
        z = np.min(xyz[:,2])

    rots = []
    for leap in range(0,360,10):
        rot = Rotation.from_euler('z', leap, degrees=True)
        rots.append(rot.as_matrix())

    best_match_points = 0
    best_tranform = np.eye(4)

    pose = np.array([x, y, z])

    threshold = 0.005
    for rot in rots:
        trans_init = np.eye(4)
        trans_init[0:3,3] = pose
        trans_init[0:3, 0:3] = rot

        reg_p2p = o3d.pipelines.registration.registration_icp(
            standard_pcd, object_pcd, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        
        # visualize_pcds( [copy.deepcopy(standard_pcd).transform(reg_p2p.transformation), object_pcd])
        match = np.array(reg_p2p.correspondence_set).shape[0]
        if( match > best_match_points):
            best_match_points = match
            best_tranform = np.array(reg_p2p.transformation)

    return best_tranform


def main():

    parser = argparse.ArgumentParser(description="extract interested object and traj from rosbag")
    parser.add_argument("-d", "--data_index", default=1,  help="Input data index.")    
    parser.add_argument("-t", "--task", default="mug_on_rack",  help="Input task name.")
    args = parser.parse_args()


    standard = np.load("mug_standard.npy", allow_pickle = True)

    test = np.load("{}_object_pcd/object_pcd_{}.npy".format(args.task, args.data_index), allow_pickle = True)
    print("current data_id: ", args.data_index)
    standard_pcd = o3d.geometry.PointCloud()
    standard_pcd.points = o3d.utility.Vector3dVector(standard)

    test_pcd = o3d.geometry.PointCloud()
    test_pcd.points = o3d.utility.Vector3dVector(test)

    # visualize_pcds( [standard_pcd, test_pcd])
    c_k = None
    F_reg = None

    init_rots = []
    found_match = False

    leap = 90
    for yaw in range (0, 360, leap):
        for pitch in range (0, 360, leap): 
            for raw in range (0, 360, leap):
                # print("r p y: ", yaw, pitch, raw)
                rot = Rotation.from_euler('zyx', [yaw, pitch, raw], degrees=True)
                init_rots.append( np.array( rot.as_matrix() ) ) 
                
    # found_match, tmp_match_points , tmp_F_reg = ICP( test, standard, init_rots[0]) # Todo, add RGB
    # visualize_pcds( [standard_pcd.transform(tmp_F_reg), test_pcd])



    source = standard_pcd

    target = test_pcd

    best_tranform = get_start_pose(test_pcd, standard_pcd, use_min_z = True)
    display_tranform = copy.deepcopy(best_tranform)
    display_tranform[2,3] += 0.15
    quat = (Rotation.from_matrix( best_tranform[0:3,0:3] ) ).as_quat()
    start_pose_7d = np.array( [ best_tranform[0][3], best_tranform[1][3], best_tranform[2][3], quat[0], quat[1], quat[2], quat[3] ] ) 
    visualize_pcds( [standard_pcd.transform(display_tranform), test_pcd])
    
    np.save("{}_start_pose/start_pose_7d_{}".format(args.task, args.data_index), start_pose_7d)


if __name__ == "__main__":
    main()

    # [frame_ids],  # we use chunk and max_episode_length to index it
    # [obs_tensors],  # wrt frame_ids, (n_cam, 2, 3, 256, 256) 
    #     obs_tensors[i][:, 0] is RGB, obs_tensors[i][:, 1] is XYZ
    # [action_tensors],  # wrt frame_ids, (1, 8)
    # [camera_dicts],
    # [gripper_tensors],  # wrt frame_ids, (1, 8) ,curretn state
    # [trajectories]  # wrt frame_ids, (N_i, 8)
    # List of tensors