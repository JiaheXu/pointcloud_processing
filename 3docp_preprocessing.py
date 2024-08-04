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
import copy
#import fpsample

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
        self.label = label
        self.n = len(self.points)
    
    # intri 3x3, extr 4x4
    def project_to_rgbd(self,
                        width,
                        height,
                        intrinsic,
                        extrinsic,
                        depth_scale,
                        depth_max
                        ):
        depth = 10.0*np.ones((height, width), dtype = float)
        depth_uint = np.zeros((height, width), dtype=np.uint16)
        color = np.zeros((height, width, 3), dtype=np.uint8)
        # xyz =  np.full((height, width, 3), np.nan)
        xyz =  np.zeros((height, width, 3), dtype = float)
        label = np.zeros((height, width), dtype=np.uint8)

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

            depth[v][u] = zc
            depth_uint[v, u] = zc * 1000
            xyz[v,u,:] = self.points[i]
            color[v, u, :] = self.colors[i] * 255
            if(label is not None):
                label[v][u] = self.label[i]

        im_color = o3d.geometry.Image(color)
        im_depth = o3d.geometry.Image(depth_uint)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            im_color, im_depth, depth_scale=1000, depth_trunc=2000, convert_rgb_to_intensity=False)
        # return rgbd
        return color, depth, xyz, label, rgbd
    
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

def visualize_pcd(pcd):
    coor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    coor_frame.scale(0.2, center=(0.,0.,0.))
    vis.add_geometry(coor_frame)
    vis.get_render_option().background_color = np.asarray([255, 255, 255])

    view_ctl = vis.get_view_control()

    vis.add_geometry(pcd)
    view_ctl.set_up((1, 0, 0))  # set the positive direction of the x-axis as the up direction
    # view_ctl.set_up((0, -1, 0))  # set the negative direction of the y-axis as the up direction
    view_ctl.set_front((-2.5, 0.0, 0.7))  # set the positive direction of the x-axis toward you
    view_ctl.set_lookat((0.0, 0.0, 0.3))  # set the original point as the center point of the window
    vis.run()
    vis.destroy_window()

def visualize_icp_result(src, dst):
    coor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    coor_frame.scale(0.2, center=(0.,0.,0.))
    vis.add_geometry(coor_frame)
    vis.get_render_option().background_color = np.asarray([255, 255, 255])

    view_ctl = vis.get_view_control()
    
    vis.add_geometry(src)
    vis.add_geometry(dst)
    view_ctl.set_up((1, 0, 0))  # set the positive direction of the x-axis as the up direction
    # view_ctl.set_up((0, -1, 0))  # set the negative direction of the y-axis as the up direction
    view_ctl.set_front((-2.5, 0.0, 0.7))  # set the positive direction of the x-axis toward you
    view_ctl.set_lookat((0.0, 0.0, 0.3))  # set the original point as the center point of the window
    vis.run()
    vis.destroy_window()

def visualize_pcd_transform(pcd, transforms, object_pcd = None):

    coor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    coor_frame.scale(0.2, center=(0.,0.,0.))
    vis.add_geometry(coor_frame)
    vis.get_render_option().background_color = np.asarray([255, 255, 255])

    view_ctl = vis.get_view_control()

    vis.add_geometry(pcd)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

    for trans in transforms:
        new_mesh = copy.deepcopy(mesh).transform(trans)
        new_mesh.scale(0.1, center=(trans[0][3], trans[1][3], trans[2][3]) )
        vis.add_geometry(new_mesh)

    if(object_pcd is not None):
        for trans in transforms:
            new_object_pcd = copy.deepcopy(object_pcd).transform(trans)
            vis.add_geometry(new_object_pcd)
    view_ctl.set_up((1, 0, 0))  # set the positive direction of the x-axis as the up direction
    view_ctl.set_front((-0.3, 0.0, 0.2))  # set the positive direction of the x-axis toward you
    view_ctl.set_lookat((0.0, 0.0, 0.3))  # set the original point as the center point of the window
    vis.run()
    vis.destroy_window()

def visualize_pcd_delta_transform(pcd, start_t, delta_transforms):

    coor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    coor_frame.scale(0.2, center=(0.,0.,0.))
    vis.add_geometry(coor_frame)
    vis.get_render_option().background_color = np.asarray([255, 255, 255])
    view_ctl = vis.get_view_control()
    if pcd is not None:
        vis.add_geometry(pcd)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    last_trans = start_t
    new_mesh = copy.deepcopy(mesh).transform(last_trans)
    new_mesh.scale(0.1, center=(last_trans[0][3], last_trans[1][3], last_trans[2][3]) )
    vis.add_geometry(new_mesh)

    for trans in delta_transforms:
        last_trans = get_transform( trans[0:3], trans[3:7] ) @ last_trans
        new_mesh = copy.deepcopy(mesh).transform(last_trans)
        new_mesh.scale(0.1, center=(last_trans[0][3], last_trans[1][3], last_trans[2][3]) )
        vis.add_geometry(new_mesh)

    view_ctl.set_up((1, 0, 0))  # set the positive direction of the x-axis as the up direction
    view_ctl.set_front((-0.3, 0.0, 0.2))  # set the positive direction of the x-axis toward you
    view_ctl.set_lookat((0.0, 0.0, 0.3))  # set the original point as the center point of the window
    vis.run()
    vis.destroy_window()

def cropping(xyz, rgb, label, bound_box):

    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    valid_idx = np.where( (x>=bound_box[0][0]) & (x <=bound_box[0][1]) & (y>=bound_box[1][0]) & (y<=bound_box[1][1]) & (z>=bound_box[2][0]) & (z<=bound_box[2][1]) )
    valid_xyz = xyz[valid_idx]
    valid_rgb = rgb[valid_idx]
    valid_label = label[valid_idx]
            
    valid_pcd = o3d.geometry.PointCloud()
    valid_pcd.points = o3d.utility.Vector3dVector( valid_xyz)
    valid_pcd.colors = o3d.utility.Vector3dVector( valid_rgb/255.0 )

    return valid_xyz, valid_rgb, valid_label, valid_pcd

def get_delta_transform(A, B): #A = d_T @ B

    d_T = A @ inv( B )
    d_rot = Rotation.from_matrix( d_T[0:3,0:3])
    d_quat = d_rot.as_quat()
    t = np.array( [ d_T[0][3], d_T[1][3], d_T[2][3], d_quat[0], d_quat[1], d_quat[2], d_quat[3] ] )
    return t


def get_start_pose(object_pcd):

    xyz = np.array( object_pcd.points )
    x = np.mean(xyz[:,0])
    y = np.mean(xyz[:,1])
    z = np.mean(xyz[:,2])
    # print("xyz: ", xyz.shape)
    # print("x: ", np.min(xyz[:,0]), np.max(xyz[:,0]))
    # print("y: ", np.min(xyz[:,1]), np.max(xyz[:,1]))
    # print("z: ", np.min(xyz[:,2]), np.max(xyz[:,2]))
    pose = [x,y,z]
    # print("pose: ", pose)
    z = np.min(xyz[:,2])
    return np.array( [ pose[0], pose[1], z, 0., 0., 0., 1.] )

def get_traj(dist, actions, transforms, traj_length):
    
    total_length = np.sum(dist)
    step_length = total_length / (traj_length - 1)
    current_step = 1.0
    traj = [transforms[0]]
    current_dist = 0.0
    for idx, action in enumerate( actions, 0):
        
        if(current_dist + dist[idx] > (current_step * step_length) - 1e-4 ):
            if(abs(current_dist - current_step * step_length) < abs(current_dist + dist[idx] - current_step * step_length) ):
                traj.append( transforms[idx] )
            else:
                traj.append( transforms[idx+1] )
            current_step = current_step + 1.0
        current_dist += dist[idx]

    return traj
def main():
    
    parser = argparse.ArgumentParser(description="extract interested object and traj from rosbag")
    # parser.add_argument("-b", "--bag_in", default="./data/yellow_handle_mug.bag",  help="Input ROS bag name.")
    parser.add_argument("-b", "--bag_in", default="./segmented_mug_on_rack1.bag",  help="Input ROS bag name.")
    
    # parser.add_argument("-b", "--bag_in", default="./segmented_mug_on_rack2.bag",  help="Input ROS bag name.")
    # parser.add_argument("-b2", "--bag_in2", default="./segmented_mug_on_rack2.bag",  help="Input ROS bag name.")
    parser.add_argument("-t", "--traj_in", default="./mug_on_rack1.npy",  help="Input trajectory file name.")
    parser.add_argument("-o", "--goal_object", default="mug", help="name of intereseted object")
    
    args = parser.parse_args()
    bagIn = rosbag.Bag(args.bag_in, "r")
    count = 0

    fxfy = 200.
    cam_intrinsic = np.array([
            [fxfy, 0., 128.0],
            [0. ,fxfy,  128.0],
            [0., 0., 1.0]
        ])
    
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(256, 256, fxfy, fxfy, 128.0, 128.0)

    cam_extrinsics = []    
    cam_extrinsics.append( get_transform( [-0.336, 0.060, 0.455], [0.653, -0.616, 0.305, -0.317]) ) # rosrun tf tf_echo  map cam1 # need to change quat
    # cam_extrinsics.append( get_transform( [0.090, 0.582, 0.449], [-0.037, 0.895, -0.443, 0.031]) ) # real world cam param
    # cam_extrinsics.append( get_transform( [0.015, -0.524, 0.448], [0.887, 0.013, 0.001, -0.461]) ) # real world cam param

    #cam_extrinsics.append( get_transform( [-0.536, 0.060, 0.455], [0.653, -0.616, 0.305, -0.317]) ) #rosrun tf tf_echo  map cam1 # need to change quat
    cam_extrinsics.append( get_transform( [-0.120, 0.582, 0.449], [-0.037, 0.895, -0.443, 0.031]) )
    cam_extrinsics.append( get_transform( [0.015, -0.624, 0.448], [0.887, 0.013, 0.001, -0.461]) )
    
    bound_box = np.array( [ [-0.2, 0.35], [ -0.6 , 0.6], [ -0.0 , 0.4] ] )

    obj_name = "mug"
    
    tmp_F_reg = np.eye(4)
    
    transform_data = np.load(args.traj_in, allow_pickle=True)
    # print("transform_data: ", len(transform_data))
    const_T = get_transform(np.array([0., 0., 0.]), np.array([0., 0., 0., 1.]) ) 
    transforms = [ cam_extrinsics[0] @ get_transform(transform_data[0][0:3], transform_data[0][3:7] ) ]
    for idx, transform in enumerate(transform_data):
        new_T = cam_extrinsics[0] @ get_transform( transform[0:3], transform[3:7] )
        if(new_T[0][3] < bound_box[0][0] or new_T[0][3] > bound_box[0][1] ):
            continue
        if(new_T[1][3] < bound_box[1][0] or new_T[1][3] > bound_box[1][1] ):
            continue
        if(new_T[2][3] < bound_box[2][0] or new_T[2][3] > bound_box[2][1] ):
            continue

        if(  np.sum( np.abs( new_T[0:3,3] - transforms[-1][0:3,3]) ) < 0.03):
            continue
        transforms.append( new_T )
    
    ros_msg = None
    idx = 0
    for topic, msg, t in bagIn.read_messages(topics=["/segmented_pointcloud"]):
        idx += 1
        if( idx != 1 ):
            continue
        ros_msg = msg # for now, just need the first one
    
    xyz, rgb, label, pcd, segmented_pointclouds = convertCloudFromRosToOpen3d( ros_msg, bound_box)

    valid_xyz, valid_rgb, valid_label, cropped_pcd = cropping( xyz, rgb, label, bound_box )
    cl, ind = cropped_pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=8.0)
    cropped_pcd = cropped_pcd.select_by_index(ind)
    valid_label = valid_label[ind]

    # env_pcd = segmented_pointclouds[0]
    object_pcd = segmented_pointclouds[1]
    # print("env_pcd: ", env_pcd)
    print("object_pcd: ", object_pcd)
    
    cl, ind = object_pcd.remove_statistical_outlier(nb_neighbors=10,std_ratio=8.0)
    object_pcd = object_pcd.select_by_index(ind)
    object_pcd = object_pcd.farthest_point_down_sample(5000)
    start_pose_6d = get_start_pose(object_pcd)



    start_pose_T = get_transform(start_pose_6d[0:3],  start_pose_6d[3:7])

    delta_T = inv( transforms[0] ) @ start_pose_T 

    # visualize_pcd_transform(cropped_pcd, transforms)

    # A_transforms = copy.deepcopy(transforms)
    A_transforms = []
    for transform in transforms:
        A_transform =  transform @ delta_T
        A_transforms.append(A_transform)

    # visualize_pcd_transform(cropped_pcd, [ transforms[0] @ delta_T, transforms[0] ])

    # visualize_pcd_transform(cropped_pcd, [start_pose_T, inv(delta_T)@start_pose_T , transforms[0]])


    # visualize_pcd_transform(cropped_pcd, transforms )
    object_pcd = object_pcd.transform( inv(start_pose_T) )
    visualize_pcd_transform(cropped_pcd, A_transforms, object_pcd )

    # visualize_pcd_transform(cropped_pcd, [start_pose_T])

    # visualize_pcd_delta_transform( cropped_pcd, delta_T, actions)

    # downpcd_farthest = uniform_down_pcd.farthest_point_down_sample(5000)

    # visualize_pcd(segmented_pointclouds[0])
    # visualize_pcd(segmented_pointclouds[1])
    # visualize_pcd(segmented_pointclouds[2])
    
    # ros_msg2 = None
    # bagIn2 = rosbag.Bag(args.bag_in2, "r")
    # idx = 0
    # for topic, msg, t in bagIn2.read_messages(topics=["/segmented_pointcloud"]):
    #     idx += 1
    #     if( idx != 1 ):
    #         continue
    #     ros_msg2 = msg # for now, just need the first one
    # xyz, rgb, label, pcd, segmented_pointclouds2 = convertCloudFromRosToOpen3d( ros_msg2, bound_box)
    # visualize_pcd(segmented_pointclouds2[0])
    # visualize_pcd(segmented_pointclouds2[1])
    # visualize_pcd(segmented_pointclouds2[2])

    # cl, ind = segmented_pointclouds[1].remove_statistical_outlier(nb_neighbors=20,std_ratio=4.0)
    # start_pcd = segmented_pointclouds[1].select_by_index(ind)
    # start_pcd = start_pcd.voxel_down_sample(voxel_size=0.003)
    # start_pcd = start_pcd.farthest_point_down_sample(600)

    # cl, ind = segmented_pointclouds2[1].remove_statistical_outlier(nb_neighbors=10,std_ratio=4.0)
    # end_pcd = segmented_pointclouds2[1].select_by_index(ind)
    # end_pcd = end_pcd.voxel_down_sample(voxel_size=0.003)
    # end_pcd = end_pcd.farthest_point_down_sample(600)

    # visualize_pcd(start_pcd)
    # visualize_pcd(end_pcd)

    # src = np.asarray(start_pcd.points)  
    # dst = np.asarray(end_pcd.points)
    # tmp_F_reg = np.eye(4)
    # print("shape: ",src.shape, " ",dst.shape)
    # found_match, tmp_match_points , tmp_F_reg = ICP( src, dst, tmp_F_reg[:3,:3], tmp_F_reg[:3,3]) # Todo, add RGB
    # result = start_pcd.transform( inv(tmp_F_reg ) )
    # visualize_icp_result( result, end_pcd)




    # whole env


    ###################################################################################### for image inputs

    # p = Projector(cropped_pcd, label)
    # for cam_idx, cam_extrinsic in enumerate(cam_extrinsics, 0):
    #     rgb, depth, xyz, label, rgbd = p.project_to_rgbd(256, 256, cam_intrinsic, inv(cam_extrinsic), 1000,10)
        
    #     data = im.fromarray(rgb) 
    #     data.save('cam{}_img{}.png'.format(cam_idx,idx))
    #     final_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    #         rgbd,
    #         o3d_intrinsic
    #     )
    #     final_pcd.transform( cam_extrinsic )
    #     # visualize_pcd(final_pcd)
    # bagIn.close()

if __name__ == "__main__":
    main()
