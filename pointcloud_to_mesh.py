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



def pcd_tp_mesh(pcd):
    # BPA(Ball-Pivoting Algorithm) Method
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)

    radius = 3 * avg_dist

    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))
    dec_mesh = bpa_mesh.simplify_quadric_decimation(100000)

    dec_mesh.remove_degenerate_triangles()
    dec_mesh.remove_duplicated_triangles()
    dec_mesh.remove_duplicated_vertices()
    dec_mesh.remove_non_manifold_edges()

    o3d.io.write_triangle_mesh("bpa_mesh.obj", dec_mesh)

    o3d.visualization.draw_geometries([dec_mesh])

    # Poisson Method
    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]

    bbox = pcd.get_axis_aligned_bounding_box()
    p_mesh_crop = poisson_mesh.crop(bbox)

    o3d.io.write_triangle_mesh("p_mesh_c.obj", p_mesh_crop)

    o3d.visualization.draw_geometries([p_mesh_crop])

def main():

    parser = argparse.ArgumentParser(description="extract interested object and traj from rosbag")
    parser.add_argument("-d", "--data_index", default=1,  help="Input data index.")    
    parser.add_argument("-t", "--task", default="mug_on_rack",  help="Input task name.")
    args = parser.parse_args()


    standard = np.load("mug_standard.npy", allow_pickle = True)

    # test = np.load("{}_object_pcd/object_p/cd_{}.npy".format(args.task, args.data_index), allow_pickle = True)
    print("current data_id: ", args.data_index)
    standard_pcd = o3d.geometry.PointCloud()
    standard_pcd.points = o3d.utility.Vector3dVector(standard)

    pcd_tp_mesh(standard_pcd)
    # visualize_pcds( [standard_pcd])
    
    # np.save("{}_start_pose/start_pose_7d_{}".format(args.task, args.data_index), start_pose_7d)


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