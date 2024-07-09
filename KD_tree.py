import numpy as np
import sys, os
import time
import glob
from collections import namedtuple
from operator import itemgetter
from pprint import pformat
import matplotlib.pyplot as plt

import argparse
from cartesian import *
from registration_3d import *

class KD_tree():
    def __init__(self, points, features = None ):
        '''
        points: numpy array (N x 3) N is number of points
        features: numpy array (N x C), C is the channel number eg: N = 3 for (rgb)
        '''
        self.N = points.shape[0] # Np: number of points
        
        
        self.features = features
        self.rectangle = np.zeros( (self.N + 10, 6) )
        self.points = np.zeros( (self.N + 10, 3) )

        self.D = 3
        
        # ls saves the index of left son
        self.ls = np.zeros( self.N + 10 )
        # rs saves the index of right son
        self.rs = np.zeros( self.N + 10 )

        self.tree = self.build(points.tolist(), 1 , self.N , depth =0 )
        self.nearest_point = []
        self.nearest_dist = 0
    
    def build( self, points, left, right, depth ):
        # build a KD-tree
        # left and right means which nodes we are looking at from 1 <= left <= right <= n
        if( left > right ):
            return 0
        
        axis = depth % self.D
        
        # sort with axis, since the number of nodes is not too big
        # we directly use O(nlogn) sort in list, rather than a O(n) sort
        points.sort( key = itemgetter(axis) )
        middle = ((left + right) // 2) 
        #print("points: ",len(points))
        #print("middle: ",middle)
        
        current_point = np.array(points[ middle - left ][0:3]).astype(float)
        self.rectangle[middle] = np.concatenate( (current_point, current_point), axis=None )
        self.points[middle] = np.array(points[ middle - left ][0:3]).astype(float)

        self.ls[ middle ] = self.build(points[:middle - left] ,left , middle-1 , depth+1 )
        self.rs[ middle ] = self.build(points[middle-left+1:]   ,middle+1, right , depth+1 )

        # after finished building son nodes, we need update father node's info 
        self.pushup(middle)
        
        return middle

    def pushup(self, root):
        # updating current node from son nodes
        # root is the current index number
        ls = self.ls[root]
        rs = self.rs[root]
        if(ls!=0):
            for i in range(3):
                self.rectangle[root,i] = min(self.rectangle[root,i],self.rectangle[int(ls),i])
                self.rectangle[root,i+3] = max(self.rectangle[root,i+3],self.rectangle[int(ls),i+3]) 
        if(rs!=0):
            for i in range(3):
                self.rectangle[root,i] = min(self.rectangle[root,i],self.rectangle[int(rs),i])
                self.rectangle[root,i+3] = max(self.rectangle[root,i+3],self.rectangle[int(rs),i+3]) 
    
    def point_to_cube(self, start_point, root):
        # compute the shortest distant from a point to a cube (L2)
        dis = np.zeros(self.D)
        
        for i in range(self.D):
            if(start_point[i] < self.rectangle[root,i]):
                dis[i] = self.rectangle[root,i] - start_point[i]
            if(start_point[i] > self.rectangle[root,i+3]):
                dis[i] = start_point[i] - self.rectangle[root,i+3]
        dist = np.linalg.norm(dis)
        return dist
    
    def find(self, start_point , left, right, depth):
        # find the closest point from start_point in a tree
        # depth tell us which dimension we should look to
        # left and right means which nodes we are looking at from 1 <=left <=right <= n
        if(left>right):
            return 
        
        middle = ((left + right) // 2) 
        
        dist = self.point_to_cube(start_point , middle)
        
        # if the current optimal solution is better than the possible solution in the cube
        # just return
        if(dist > self.nearest_dist):
            return
        
        # check the distance from start_point to the current node's distance
        tmp = self.points[middle]
        dist = np.linalg.norm(start_point - tmp)
        
        if( dist < self.nearest_dist):
            self.nearest_dist = dist
            self.nearest_point = tmp
        
        # look into son nodes
        self.find( start_point , left , middle-1 ,depth)
        self.find( start_point , middle+1 , right,depth)
        
    def FindClosestPoint(self, start_point ):
        
        self.nearest_dist =  np.finfo(np.float32).max
    
        self.find( start_point , 1 , self.N , depth=0 ) 
        
        return self.nearest_point
    


def ICP( src, dst, init_rot = None, init_trans = None, brute_force = False ):

    start = time.time()
    F_reg = np.eye(4)

    if init_rot is not None:
        F_reg[0:3,0:3] = init_rot

    if init_trans is not None:
        init_trans = init_trans.reshape(3,)
        F_reg[0:3,3] = init_trans


    kdtree = None
    if(brute_force == False):
        kdtree = KD_tree(src)
    
    Ns = dst.shape[0]
    
    closest_p = []
    iteration = 80
    c_k = []
    old_pts = []
    
    closest_p = []
    
    same_point_threshold = 0.005
    
    found_match = False
    match_points = 0

    for i in range(iteration):
        print("ICP in iter: ", i)
        s_k_i = points_transform(F_reg , dst) 
        
        closest_p = []
        match_points = 0

        for j in range(Ns):
            min_dist = np.finfo(np.float32).max
            tmp = kdtree.FindClosestPoint( s_k_i[j] )
            closest_p.append(tmp)
            dist = np.linalg.norm(s_k_i[j] - tmp)
            if( dist < same_point_threshold): # Todo add RGB feature
                match_points += 1

        print("match_points: ", match_points, " / ", Ns)
        c_k = np.array(closest_p)
        
        if(i == int( iteration * 0.5) and  match_points <= int( 0.5 * Ns ) ):
            return found_match, match_points , F_regNew
        
        # Update guess
        old_pts = c_k

        deltaF_reg = registration_3d( s_k_i , c_k)

        F_regNew = deltaF_reg @ F_reg
        run_time = time.time() - start
        if np.sum(np.abs(F_reg - F_regNew)) < 1e-4:
            print("The results for has been found after "+ str(i) + " iterations!")
            print("Runtime is " + str(run_time) + " seconds.")
            return found_match, match_points , F_regNew

        F_reg = F_regNew

        print("np.linalg.det(rot): ", np.linalg.det(F_reg[0:3, 0:3]) )

        if(np.linalg.det(F_reg[0:3, 0:3]) < 0.8):
            print("invalid init")
            return found_match, match_points , F_regNew
        if i == iteration - 1:
            print("MAX_ITER reached without finding stable results!")
        
        if match_points > int( 0.7 * Ns ):
            found_match = True

            
    return found_match, match_points , F_reg
