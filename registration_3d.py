import numpy as np
from cartesian import *
######
# calculate the lsq transform between 2 sets 
# start and goal are a nx3 numpy array 
# return a 4*4 transform matrix F that F*start[i] = goal[i] in a lsq way 
######
def registration_3d( start , goal ):
    n = np.shape(start)[0]
    #middle point
    start_mid = np.sum(start, axis=0)/n
    start = start - start_mid
    
    goal_mid = np.sum(goal, axis=0)/n
    goal = goal - goal_mid
    
    # according to formula on rigid3d3dcalculations.pdf p9
    H = start.T @ goal
    u,s,vt = np.linalg.svd(H)
    v = vt.T
    R = v@(u.T)
    if np.abs( np.linalg.det(R) - 1 ) > 1e-5: #det(R) != 1
        #A@t = b
        #use lsq find t, lsq method have relatively greater error, so we take it as a plan B
        A = np.zeros((3*n, 9))
        B = np.zeros((3*n, 1))
        for i in range(n):
            A[i*3,0] = start[i,0]
            A[i*3,1] = start[i,1]
            A[i*3,2] = start[i,2]
            A[i*3+1,3] = start[i,0]
            A[i*3+1,4] = start[i,1]
            A[i*3+1,5] = start[i,2]
            A[i*3+2,6] = start[i,0]
            A[i*3+2,7] = start[i,1]
            A[i*3+2,8] = start[i,2]
            
            B[i*3] = goal[i,0]
            B[i*3+1] = goal[i,1]
            B[i*3+2] = goal[i,2]
            
        X = np.linalg.lstsq(A, B, rcond=None)[0]
        R = X.reshape(3,3)
        
    p = goal_mid - R@start_mid

    F = concat_frame(R,p)
    return F
