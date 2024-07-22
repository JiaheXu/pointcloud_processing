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
from matplotlib import pyplot as plt
import copy

import rospy
import rosbag
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from numpy.linalg import inv
from lib_cloud_conversion_between_Open3D_and_ROS import convertCloudFromRosToOpen3d
from scipy.spatial.transform import Rotation

# ICP
from KD_tree import *
from numpy.lib.function_base import diff
from registration_3d import *
from cartesian import *
colors = ['tab:red' ,'tab:blue', 'tab:orange', 'tab:green']

class Projector:
    def __init__(self, cloud, label = None) -> None:
        self.cloud = cloud
        self.points = np.asarray(cloud.points)
        self.colors = np.asarray(cloud.colors)
        self.n = len(self.points)
        self.label = label

    # intri 3x3, extr 4x4
    def project_to_rgbd(self,
                        width,
                        height,
                        intrinsic,
                        extrinsic,
                        depth_scale,
                        depth_max
                        ):
        depth = 10.0*np.ones((height, width, 1), dtype = float)
        color = np.zeros((height, width, 3), dtype=np.uint8)

        # object order works, but not general
        # for obj in range(0,3):
        #     obj_idxs = np.where(self.label == obj)[0]
        #     for i in obj_idxs:
        #         print(i)
        #         point4d = np.append(self.points[i], 1)
        #         new_point4d = np.matmul(extrinsic, point4d)
        #         point3d = new_point4d[:-1]
        #         zc = point3d[2]
        #         new_point3d = np.matmul(intrinsic, point3d)
        #         new_point3d = new_point3d/new_point3d[2]
        #         u = int(round(new_point3d[0]))
        #         v = int(round(new_point3d[1]))

        #         # Fixed u, v checks. u should be checked for width
        #         if (u < 0 or u > width - 1 or v < 0 or v > height - 1 or zc <= 0 or zc > depth_max):
        #             continue
        #         if(zc > depth[v][u]):
        #             continue

        #         depth[v, u ] = float(zc)
        #         color[v, u, :] = self.colors[i] * 255



        for i in range(0, self.n):
            point4d = np.append(self.points[i], 1)
            new_point4d = np.matmul(extrinsic, point4d)
            point3d = new_point4d[:-1]
            zc = point3d[2]
            new_point3d = np.matmul(intrinsic, point3d)
            new_point3d = new_point3d/new_point3d[2]
            u = int(round(new_point3d[0]))
            v = int(round(new_point3d[1]))

            # Fixed u, v checks. u should be checked for width
            if (u < 0 or u > width - 1 or v < 0 or v > height - 1 or zc <= 0.0 or zc > depth_max):
                continue
            if(zc > depth[v][u]):
                continue

            depth[v, u ] = float(zc)
            color[v, u, :] = self.colors[i] * 255

        return color, depth


def plot_func(src, dst, dir="", idx=0, save = False):

    if(src.shape[0] != 3):
        src = np.transpose(src)
    if(dst.shape[0] != 3):
        dst = np.transpose(dst)    
    fig = plt.figure()
    ax = fig.add_subplot(111,projection="3d")

    ax.scatter(src[0, :], src[1, :], src[2, :], c = colors[0])
    ax.scatter(dst[0, :], dst[1, :], dst[2, :], c = colors[1])
    if(save == False):
        plt.show()
    else:
        plt.savefig(dir + "/" + str(idx) + ".svg")

def get_init_trans(src, dst):
    if(src.shape[0] == 3):
        src = np.transpose(src)
    if(dst.shape[0] == 3):
        dst = np.transpose(dst)
    trans = np.mean(src, axis = 0) - np.mean(dst, axis = 0)
    trans = trans.reshape(3,1)
    return trans


def print_plot(F_reg, src, dst, dir, idx = 0,  save = False):
    rot = F_reg[0:3, 0:3]
    trans = F_reg[0:3, 3]
    
    trans = trans.reshape(3,1)
    print("rot: ", rot)
    print("trans: ", trans)
    pcd = rot @ np.transpose(dst) + trans
    plot_func( src, pcd , dir, idx = idx, save = save)
    
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud])

def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target],
                                      zoom=0.5,
                                      front=[-0.2458, -0.8088, 0.5342],
                                      lookat=[1.7745, 2.2305, 0.9787],
                                      up=[0.3109, -0.5878, -0.7468])

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def colored_ICP(source, target):
    voxel_radius = [0.002, 0.002, 0.002]
    max_iter = [50, 30, 14]
    current_transformation = np.identity(4)
    print("3. Colored point cloud registration")
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print([iter, radius, scale])

        print("3-1. Downsample with a voxel size %.2f" % radius)
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        print("3-2. Estimate normal.")
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        print("3-3. Applying colored point cloud registration")
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                            relative_rmse=1e-6,
                                                            max_iteration=iter))
        current_transformation = result_icp.transformation
        print(result_icp)
    draw_registration_result_original_color(source, target,
                                            result_icp.transformation)

# def ICP(source, target):

#     threshold = 0.02
#     trans_init = np.eye(4) 
#     reg_p2p = o3d.pipelines.registration.registration_icp(
#         source, target, threshold, trans_init,
#         o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#         criteria=o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-06, relative_rmse=1e-06, max_iteration=2000)
#         )
#     print(reg_p2p)
#     print("Transformation is:")
#     print(reg_p2p.transformation)
#     draw_registration_result(source, target, reg_p2p.transformation)

def get_transform( trans, quat):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat( quat ).as_matrix()
    t[:3, 3] = trans
    return t

def get_cube_corners( bound_box ):
    corners = []
    corners.append( [ bound_box[0][0], bound_box[1][0], bound_box[2][0] ])
    corners.append( [ bound_box[0][0], bound_box[1][1], bound_box[2][0] ])
    corners.append( [ bound_box[0][1], bound_box[1][1], bound_box[2][0] ])
    corners.append( [ bound_box[0][1], bound_box[1][0], bound_box[2][0] ])

    return corners

def main():
    
    parser = argparse.ArgumentParser(description="extract interested object and traj from rosbag")
    # parser.add_argument("-b", "--bag_in", default="./data/yellow_handle_mug.bag",  help="Input ROS bag name.")
    parser.add_argument("-b", "--bag_in", default="./segmented_mug_traj1.bag",  help="Input ROS bag name.")
    parser.add_argument("-o", "--goal_object", default="mug", help="name of intereseted object")
    # parser.add_argument("-b", "--bag_in", default="traj1.bag",  help="Input ROS bag name.")
    
    args = parser.parse_args()
    bagIn = rosbag.Bag(args.bag_in, "r")
    count = 0
    
    # cam_intrinsic = np.array([
    #         [490, 0., 640],
    #         [0. ,490,  360],
    #         [0., 0., 1.0]
    #     ])
    
    # cam_intrinsic = np.array([
    #         [245, 0., 320],
    #         [0. ,245,  180],
    #         [0., 0., 1.0]
    #     ])

    cam_intrinsic = np.array([
            [80, 0., 128],
            [0. ,80,  128],
            [0., 0., 1.0]
        ])

    cam_extrinsics = []
    
    cam_extrinsics.append( get_transform( [-0.336, 0.060, 0.455], [0.653, -0.616, 0.305, -0.317]) ) #rosrun tf tf_echo  map cam1
    cam_extrinsics.append( get_transform( [0.090, 0.582, 0.449], [-0.037, 0.895, -0.443, 0.031]) )
    cam_extrinsics.append( get_transform( [0.015, -0.524, 0.448], [0.887, 0.013, 0.001, -0.461]) )
    
    # cam_extrinsic = inv(cam_extrinsics[0])

    # min_bound = np.array([-0.2, -1.0, 0.0])
    # max_bound = np.array([ 0.4,  1.0, 0.6])
    bound_box = np.array( [ [-0.2, 0.35], [ -1.0 , 1.0], [ -0.1 , 0.4] ] )
    vol = o3d.visualization.SelectionPolygonVolume()
    vol.orthogonal_axis = "Z"
    vol.axis_max = 1.0
    vol.axis_min = -1.0
    ply_point_cloud = o3d.data.PLYPointCloud()
    object_pcd = o3d.io.read_point_cloud("./mug1.ply")

    # object_pcd = object_pcd.uniform_down_sample( every_k_points=5 )
    # print(pcd)
    # print(np.asarray(pcd.points))
    # o3d.visualization.draw_geometries([object_pcd])
    corners = get_cube_corners(bound_box)
    # for i in range(0,2):
    #     for j in range(0,2):
    #         for k in range(0,2):       
    #             corners.append( [ bound_box[0][i], bound_box[1][j], bound_box[2][k] ] )
    corners = np.array(corners)
    bounding_polygon = corners.astype("float64")

    obj_name = "mug"
    idx = 0
    tmp_F_reg = np.eye(4)

    for topic, msg, t in bagIn.read_messages(topics=["/segmented_pointcloud"]):
        idx += 1
        if( idx != 1):
            continue
        pcd, label, segmented_pointclouds = convertCloudFromRosToOpen3d( msg )
        o3d.visualization.draw_geometries([pcd])
  
        

        vol.bounding_polygon = o3d.utility.Vector3dVector(bounding_polygon)
        cropped_pcd = vol.crop_point_cloud(pcd)
        o3d.visualization.draw_geometries([cropped_pcd])

        # p = Projector(pcd, label)

        # for cam_idx, cam_extrinsic in enumerate(cam_extrinsics,0):
        #     rgb, depth = p.project_to_rgbd(256, 256, cam_intrinsic, inv(cam_extrinsic), 1000,10)
        #     data = im.fromarray(rgb) 
        #     data.save('cam{}_img{}.png'.format(cam_idx,idx)) 

        # for pcd in segmented_pointclouds:
            # print(pcd)

        # o3d.visualization.draw_geometries([segmented_pointclouds[1]])

        # downpcd = segmented_pointclouds[1].uniform_down_sample( every_k_points=10 )
        # o3d.io.write_point_cloud("moved_mug1.ply", downpcd)
        # src = np.asarray(object_pcd.points)
        # dst = np.asarray(downpcd.points)
        # print("shape: ",src.shape, " ",dst.shape)
        # found_match, tmp_match_points , tmp_F_reg = ICP( src, dst , tmp_F_reg[:3,:3], tmp_F_reg[:3,3]) # Todo, add RGB
        # print_plot(tmp_F_reg, src, dst, dir = obj_name + "/" + (args.bag_in )[:-4],  idx=idx, save = True)



    print(count)

    bagIn.close()

if __name__ == "__main__":
    main()