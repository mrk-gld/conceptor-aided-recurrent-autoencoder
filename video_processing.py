import os, sys

sys.path.append('AMCParser')
import amc_parser as amc

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import transforms3d 
from matplotlib.animation import FuncAnimation
import numpy as npy
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg 
from typing import NamedTuple, Callable, Optional

import functools
from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D
import transforms3d 

from sklearn.preprocessing import MinMaxScaler
from numpy import Inf, NaN

from skeleton import angle2Coordinate

def rotate_translate_frame(points, rotation, translation_hrz):
    # translate to put back at origin and rotate to be in the local frame
    points_rot_transl = rotation.dot(points - translation_hrz)
    return points_rot_transl


def inv_rotate_translate_frame(points, rotation, translation_hrz):
    points_rot_transl = rotation.T.dot(points) + translation_hrz  
    return points_rot_transl


def vector2d_angles(v1, v2):
    """ Returns the angle between the two 2d vectors. Faster than computing the rotation matrix between the two vectors 
    and extracting the angle from it. It's also easier, because : there's no readily available vector to matrix conversion function and 
    there's many ways/conventions to do it.
    
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    cross = np.cross(v1, v2)
    dot = np.dot(v1, v2)
    return np.arctan2(cross, dot)


def video_generation(joint_coordinate_arr, c_joints, out_path = 'simple_animation.gif', display = "normal"): # type: normal, treadmill, circle
    fig = plt.figure(figsize=(10, 10))
    # ax = Axes3D(fig)
    ax = fig.add_subplot(111, projection="3d")


    def draw_frame(i):
        ax.cla()
        ax.set_xlim3d(-50, 10)
        ax.set_ylim3d(-20, 40)
        ax.set_zlim3d(-20, 40)
        joint_coordinate = joint_coordinate_arr[i].copy()
        if display == "treadmill":
            offcet = joint_coordinate['root'].copy()
            offcet[1] = 0
            for joint_coord in joint_coordinate.values():
                joint_coord -= offcet
            ax.plot(offcet[2], offcet[0], offcet[1], 'g.', markersize=10)
        
        # if display == "circle":
        #     offcet = joint_coordinate['root'].copy()
        #     offcet[1] = 0
        #     # if it goes beyon the boundary it appears on the other side
        #     circle_coord = offcet.copy()
        #     if offcet[2] > 10:
        #         circle_coord[2] = -50
        #     if circle_coord[2] < -50:
            # for joint_coord in joint_coordinate.values():
            #     joint_coord -= offcet + circle_coord
            # ax.plot(offcet[2], offcet[0], offcet[1], 'g.', markersize=10)
        
        # joints['root'].set_motion(motions[i])
        # c_joints = joints['root'].to_dict()
        xs, ys, zs = [], [], []
        for joint_coord in joint_coordinate.values():
            xs.append(joint_coord[0])
            ys.append(joint_coord[1])
            zs.append(joint_coord[2])
            ax.plot(zs, xs, ys, 'b.')

        for joint in c_joints.values():
            child = joint
            if child.parent is not None:
                parent = child.parent
                xs = [joint_coordinate[child.name][0], joint_coordinate[parent.name][0]]
                ys = [joint_coordinate[child.name][1], joint_coordinate[parent.name][1]]
                zs = [joint_coordinate[child.name][2], joint_coordinate[parent.name][2]]
                ax.plot(zs, xs, ys, 'r')
    FuncAnimation(fig, draw_frame, range(0, len(joint_coordinate_arr)-1, 10)).save(out_path, 
                                                    bitrate=8000,
                                                    fps=8)
    plt.close('all')
    
    
def convert_motion_data_to_video(datasets,data_mean,data_std,labels):
    
    # unnormalize the data
    for i, data in enumerate(datasets):
        datasets[i] = data*data_std + data_mean

    data_concat = npy.concatenate(datasets, axis=0)

    scaler = MinMaxScaler(feature_range=(-1, 1))

    scaler = MinMaxScaler(feature_range=(-1, 1))
    # data_concat_sc_sh = scaler.fit_transform(data_concat)
    
    #check if all data sets have the same length
    len_data = [len(data) for data in datasets]
    min_len = min(len_data)
    for i, data in enumerate(datasets):
        datasets[i] = data[:min_len]

    # load the initial condition for the movement
    data_ini_walk = np.load("motion_data_numpy/data_ini_walk.npy")
    data_ini_walk = [[data_ini_walk[0], data_ini_walk[1]], [data_ini_walk[2]]]

    # load the information about the skeleton
    c_joints = np.load('motion_data_numpy/c_joint.npy', allow_pickle=True).item()
    norm_ij = np.load('motion_data_numpy/norm_ij.npy')

    ## video of the behavior
    # data_interp_unsc = scaler.inverse_transform(data_interpolated)
    for i, data in enumerate(datasets):
        joint_coordinate_arr = angle2Coordinate(data, data_ini_walk, c_joints, norm_ij)
        video_generation(joint_coordinate_arr, c_joints, out_path = f'./{labels[i]}.mp4', display = "treadmill")