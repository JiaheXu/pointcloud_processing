"""
pcd_obs_env with:
1. object/background segmentation
2. object registration
3. goal sampling
4. reward calculation
"""

import numpy as np

import os
import argparse
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import open3d as o3d
import numpy as np
from ctypes import * # convert float to uint32

import rospy
import rosbag
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

from lib_cloud_conversion_between_Open3D_and_ROS import convertCloudFromRosToOpen3d
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
def main():
    
    parser = argparse.ArgumentParser(description="extract interested object and traj from rosbag")
    parser.add_argument("-b", "--bag_in", default="./data/yellow_handle_mug.bag",  help="Input ROS bag name.")
    parser.add_argument("-o", "--goal_object", default="mug", help="name of intereseted object")
    # parser.add_argument("-b", "--bag_in", default="traj1.bag"  help="Input ROS bag name.")
    
    args = parser.parse_args()
    bagIn = rosbag.Bag(args.bag_in, "r")


    count = 0
    for topic, msg, t in bagIn.read_messages(topics=["/object_point_cloud2"]):
        pcd = convertCloudFromRosToOpen3d( msg )
        # print(o3d_pcd)
        downpcd = pcd.voxel_down_sample(voxel_size=0.005)
        print("downpcd: ", downpcd)
        cl, ind = downpcd.remove_radius_outlier(nb_points=8, radius=0.008)
        display_inlier_outlier(downpcd, ind)
        # downpcd = pcd.uniform_down_sample( every_k_points=5 )
        # o3d.visualization.draw_geometries([downpcd])
        count += 1

    print(count)

    bagIn.close()

if __name__ == "__main__":
    main()