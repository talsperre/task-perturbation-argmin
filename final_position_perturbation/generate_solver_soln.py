import os
import time
import json
import autograd
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from copy import deepcopy
from scipy import optimize
from tqdm.notebook import tqdm
from transforms3d import euler
from autograd import grad, jacobian
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize, Bounds
from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import autograd.numpy as np

from jax import jit, jacfwd, jacrev, random, vmap

parser = argparse.ArgumentParser(description="Config parser")
parser.add_argument("--traj_num", help="Current trajectory number", type=int)
parser.add_argument("--dataset_path", help="Current trajectory number", type=str)
parser.add_argument("--results_path", help="Directory to save plots and trajectories", type=str)
args = parser.parse_args()

### Intializations
np.random.seed(42)
num = 50
rho_vel = 100
rho_acc = rho_vel * 10
rho_b = rho_vel * 10
rho_jerk = rho_vel * 100
weight_pos = 100.0
weight_orient = 10.0
roll_des = - 87.2 * np.pi / 180
pitch_des = - 41.0 * np.pi / 180

q_min = np.array([-165, -100, -165, -165, -165, -1.0, -165]) * np.pi / 180
q_max = np.array([ 165,  101,  165,  1.0,  165,  214,  165]) * np.pi / 180

q_min_traj = np.hstack((q_min[0] * np.ones(num), q_min[1] * np.ones(num), q_min[2] * np.ones(num), q_min[3] * np.ones(num), q_min[4] * np.ones(num), q_min[5] * np.ones(num), q_min[6] * np.ones(num)))
q_max_traj = np.hstack((q_max[0] * np.ones(num), q_max[1] * np.ones(num), q_max[2] * np.ones(num), q_max[3] * np.ones(num), q_max[4] * np.ones(num), q_max[5] * np.ones(num), q_max[6] * np.ones(num)))

A = np.identity(num)
A_vel = np.diff(A, axis=0)
A_acc = np.diff(A_vel, axis=0)
A_jerk = np.diff(A_acc, axis=0)

cur_pos = args.traj_num
data = np.load(args.dataset_path, allow_pickle=True)
data = data.item()[cur_pos]

#### Initial and final positions of the manipulator
q_init = data['q_init']
# Choose any one of these final positions
q_fin = data['q_fin'][0]

q_1_init, q_2_init, q_3_init, q_4_init, q_5_init, q_6_init, q_7_init = q_init
q_1_fin, q_2_fin, q_3_fin, q_4_fin, q_5_fin, q_6_fin, q_7_fin = q_fin

#### Transforms
def get_transforms(q_1, q_2, q_3, q_4, q_5, q_6, q_7):
    x = -0.107 * (((np.sin(q_1) * np.sin(q_3) - np.cos(q_1) * np.cos(q_2) * np.cos(q_3)) * np.cos(q_4) - np.sin(q_2) * np.sin(q_4) * np.cos(q_1)) * np.cos(q_5) + (np.sin(q_1) * np.cos(q_3) + np.sin(q_3) * np.cos(q_1) * np.cos(q_2)) * np.sin(q_5)) * np.sin(q_6) - 0.088 * (((np.sin(q_1) * np.sin(q_3) - np.cos(q_1) * np.cos(q_2) * np.cos(q_3)) * np.cos(q_4) - np.sin(q_2) * np.sin(q_4) * np.cos(q_1)) * np.cos(q_5) + (np.sin(q_1) * np.cos(q_3) + np.sin(q_3) * np.cos(q_1) * np.cos(q_2)) * np.sin(q_5)) * np.cos(q_6) + 0.088 * ((np.sin(q_1) * np.sin(q_3) - np.cos(q_1) * np.cos(q_2) * np.cos(q_3)) * np.sin(q_4) + np.sin(q_2) * np.cos(q_1) * np.cos(q_4)) * np.sin(q_6) - 0.107 * ((np.sin(q_1) * np.sin(q_3) - np.cos(q_1) * np.cos(q_2) * np.cos(q_3)) * np.sin(q_4) + np.sin(q_2) * np.cos(q_1) * np.cos(q_4)) * np.cos(q_6) + 0.384 * (np.sin(q_1) * np.sin(q_3) - np.cos(q_1) * np.cos(q_2) * np.cos(q_3)) * np.sin(q_4) + 0.0825 * (np.sin(q_1) * np.sin(q_3) - np.cos(q_1) * np.cos(q_2) * np.cos(q_3)) * np.cos(q_4) - 0.0825 * np.sin(q_1) * np.sin(q_3) - 0.0825 * np.sin(q_2) * np.sin(q_4) * np.cos(q_1) + 0.384 * np.sin(q_2) * np.cos(q_1) * np.cos(q_4) + 0.316 * np.sin(q_2) * np.cos(q_1) + 0.0825 * np.cos(q_1) * np.cos(q_2) * np.cos(q_3)
    y = 0.107 * (((np.sin(q_1) * np.cos(q_2) * np.cos(q_3) + np.sin(q_3) * np.cos(q_1)) * np.cos(q_4) + np.sin(q_1) * np.sin(q_2) * np.sin(q_4)) * np.cos(q_5) - (np.sin(q_1) * np.sin(q_3) * np.cos(q_2) - np.cos(q_1) * np.cos(q_3)) * np.sin(q_5)) * np.sin(q_6) + 0.088 * (((np.sin(q_1) * np.cos(q_2) * np.cos(q_3) + np.sin(q_3) * np.cos(q_1)) * np.cos(q_4) + np.sin(q_1) * np.sin(q_2) * np.sin(q_4)) * np.cos(q_5) - (np.sin(q_1) * np.sin(q_3) * np.cos(q_2) - np.cos(q_1) * np.cos(q_3)) * np.sin(q_5)) * np.cos(q_6) - 0.088 * ((np.sin(q_1) * np.cos(q_2) * np.cos(q_3) + np.sin(q_3) * np.cos(q_1)) * np.sin(q_4) - np.sin(q_1) * np.sin(q_2) * np.cos(q_4)) * np.sin(q_6) + 0.107 * ((np.sin(q_1) * np.cos(q_2) * np.cos(q_3) + np.sin(q_3) * np.cos(q_1)) * np.sin(q_4) - np.sin(q_1) * np.sin(q_2) * np.cos(q_4)) * np.cos(q_6) - 0.384 * (np.sin(q_1) * np.cos(q_2) * np.cos(q_3) + np.sin(q_3) * np.cos(q_1)) * np.sin(q_4) - 0.0825 * (np.sin(q_1) * np.cos(q_2) * np.cos(q_3) + np.sin(q_3) * np.cos(q_1)) * np.cos(q_4) - 0.0825 * np.sin(q_1) * np.sin(q_2) * np.sin(q_4) + 0.384 * np.sin(q_1) * np.sin(q_2) * np.cos(q_4) + 0.316 * np.sin(q_1) * np.sin(q_2) + 0.0825 * np.sin(q_1) * np.cos(q_2) * np.cos(q_3) + 0.0825 * np.sin(q_3) * np.cos(q_1)
    z = -0.107 * ((np.sin(q_2) * np.cos(q_3) * np.cos(q_4) - np.sin(q_4) * np.cos(q_2)) * np.cos(q_5) - np.sin(q_2) * np.sin(q_3) * np.sin(q_5)) * np.sin(q_6) - 0.088 * ((np.sin(q_2) * np.cos(q_3) * np.cos(q_4) - np.sin(q_4) * np.cos(q_2)) * np.cos(q_5) - np.sin(q_2) * np.sin(q_3) * np.sin(q_5)) * np.cos(q_6) + 0.088 * (np.sin(q_2) * np.sin(q_4) * np.cos(q_3) + np.cos(q_2) * np.cos(q_4)) * np.sin(q_6) - 0.107 * (np.sin(q_2) * np.sin(q_4) * np.cos(q_3) + np.cos(q_2) * np.cos(q_4)) * np.cos(q_6) + 0.384 * np.sin(q_2) * np.sin(q_4) * np.cos(q_3) + 0.0825 * np.sin(q_2) * np.cos(q_3) * np.cos(q_4) - 0.0825 * np.sin(q_2) * np.cos(q_3) - 0.0825 * np.sin(q_4) * np.cos(q_2) + 0.384 * np.cos(q_2) * np.cos(q_4) + 0.316 * np.cos(q_2) + 0.33
    tr = np.sin(q_2) * np.sin(q_3) * np.sin(q_5) * np.sin(q_6) + np.sin(q_2) * np.sin(q_4) * np.sin(q_5) * np.sin(q_1 + q_7) - np.sin(q_2) * np.sin(q_4) * np.cos(q_3) * np.cos(q_6) + np.sin(q_2) * np.sin(q_4) * np.cos(q_5) * np.cos(q_6) * np.cos(q_1 + q_7) - np.sin(q_2) * np.sin(q_6) * np.cos(q_3) * np.cos(q_4) * np.cos(q_5) + np.sin(q_2) * np.sin(q_6) * np.cos(q_4) * np.cos(q_1 + q_7) + np.sin(q_3) * np.sin(q_4) * np.sin(q_6) * np.sin(q_1 + q_7) - np.sin(q_3) * np.sin(q_5) * np.cos(q_2) * np.cos(q_6) * np.cos(q_1 + q_7) + np.sin(q_3) * np.sin(q_5) * np.cos(q_4) * np.cos(q_1 + q_7) + np.sin(q_3) * np.sin(q_1 + q_7) * np.cos(q_2) * np.cos(q_5) - np.sin(q_3) * np.sin(q_1 + q_7) * np.cos(q_4) * np.cos(q_5) * np.cos(q_6) - np.sin(q_4) * np.sin(q_6) * np.cos(q_2) * np.cos(q_3) * np.cos(q_1 + q_7) + np.sin(q_4) * np.sin(q_6) * np.cos(q_2) * np.cos(q_5) + np.sin(q_5) * np.sin(q_1 + q_7) * np.cos(q_2) * np.cos(q_3) * np.cos(q_4) - np.sin(q_5) * np.sin(q_1 + q_7) * np.cos(q_3) * np.cos(q_6) + np.cos(q_2) * np.cos(q_3) * np.cos(q_4) * np.cos(q_5) * np.cos(q_6) * np.cos(q_1 + q_7) - np.cos(q_2) * np.cos(q_4) * np.cos(q_6) - np.cos(q_3) * np.cos(q_5) * np.cos(q_1 + q_7)
    ax1 = -0.5 * (((np.sin(q_1) * np.cos(q_2) * np.cos(q_3) + np.sin(q_3) * np.cos(q_1)) * np.cos(q_4) + np.sin(q_1) * np.sin(q_2) * np.sin(q_4)) * np.cos(q_5) - (np.sin(q_1) * np.sin(q_3) * np.cos(q_2) - np.cos(q_1) * np.cos(q_3)) * np.sin(q_5)) * np.sin(q_6) + 0.5 * (((np.sin(q_2) * np.cos(q_3) * np.cos(q_4) - np.sin(q_4) * np.cos(q_2)) * np.cos(q_5) - np.sin(q_2) * np.sin(q_3) * np.sin(q_5)) * np.cos(q_6) - (np.sin(q_2) * np.sin(q_4) * np.cos(q_3) + np.cos(q_2) * np.cos(q_4)) * np.sin(q_6)) * np.sin(q_7) - 0.5 * ((np.sin(q_1) * np.cos(q_2) * np.cos(q_3) + np.sin(q_3) * np.cos(q_1)) * np.sin(q_4) - np.sin(q_1) * np.sin(q_2) * np.cos(q_4)) * np.cos(q_6) - 0.5 * ((np.sin(q_2) * np.cos(q_3) * np.cos(q_4) - np.sin(q_4) * np.cos(q_2)) * np.sin(q_5) + np.sin(q_2) * np.sin(q_3) * np.cos(q_5)) * np.cos(q_7)
    ax2 = -0.5 * (((np.sin(q_1) * np.sin(q_3) - np.cos(q_1) * np.cos(q_2) * np.cos(q_3)) * np.cos(q_4) - np.sin(q_2) * np.sin(q_4) * np.cos(q_1)) * np.cos(q_5) + (np.sin(q_1) * np.cos(q_3) + np.sin(q_3) * np.cos(q_1) * np.cos(q_2)) * np.sin(q_5)) * np.sin(q_6) + 0.5 * (((np.sin(q_2) * np.cos(q_3) * np.cos(q_4) - np.sin(q_4) * np.cos(q_2)) * np.cos(q_5) - np.sin(q_2) * np.sin(q_3) * np.sin(q_5)) * np.cos(q_6) - (np.sin(q_2) * np.sin(q_4) * np.cos(q_3) + np.cos(q_2) * np.cos(q_4)) * np.sin(q_6)) * np.cos(q_7) - 0.5 * ((np.sin(q_1) * np.sin(q_3) - np.cos(q_1) * np.cos(q_2) * np.cos(q_3)) * np.sin(q_4) + np.sin(q_2) * np.cos(q_1) * np.cos(q_4)) * np.cos(q_6) + 0.5 * ((np.sin(q_2) * np.cos(q_3) * np.cos(q_4) - np.sin(q_4) * np.cos(q_2)) * np.sin(q_5) + np.sin(q_2) * np.sin(q_3) * np.cos(q_5)) * np.sin(q_7)
    ax3 = -0.5 * np.sin(q_2) * np.sin(q_4) * np.sin(q_5) * np.cos(q_1 + q_7) + 0.5 * np.sin(q_2) * np.sin(q_4) * np.sin(q_1 + q_7) * np.cos(q_5) * np.cos(q_6) + 0.5 * np.sin(q_2) * np.sin(q_6) * np.sin(q_1 + q_7) * np.cos(q_4) - 0.5 * np.sin(q_3) * np.sin(q_4) * np.sin(q_6) * np.cos(q_1 + q_7) - 0.5 * np.sin(q_3) * np.sin(q_5) * np.sin(q_1 + q_7) * np.cos(q_2) * np.cos(q_6) + 0.5 * np.sin(q_3) * np.sin(q_5) * np.sin(q_1 + q_7) * np.cos(q_4) - 0.5 * np.sin(q_3) * np.cos(q_2) * np.cos(q_5) * np.cos(q_1 + q_7) + 0.5 * np.sin(q_3) * np.cos(q_4) * np.cos(q_5) * np.cos(q_6) * np.cos(q_1 + q_7) - 0.5 * np.sin(q_4) * np.sin(q_6) * np.sin(q_1 + q_7) * np.cos(q_2) * np.cos(q_3) - 0.5 * np.sin(q_5) * np.cos(q_2) * np.cos(q_3) * np.cos(q_4) * np.cos(q_1 + q_7) + 0.5 * np.sin(q_5) * np.cos(q_3) * np.cos(q_6) * np.cos(q_1 + q_7) + 0.5 * np.sin(q_1 + q_7) * np.cos(q_2) * np.cos(q_3) * np.cos(q_4) * np.cos(q_5) * np.cos(q_6) - 0.5 * np.sin(q_1 + q_7) * np.cos(q_3) * np.cos(q_5)
    return x, y, z, tr, ax1, ax2, ax3

def fk_franka(q):
    q_1 = q[0]
    q_2 = q[1]
    q_3 = q[2]
    q_4 = q[3]
    q_5 = q[4]
    q_6 = q[5]
    q_7 = q[6]
    x, y, z, tr, ax1, ax2, ax3 = get_transforms(q_1, q_2, q_3, q_4, q_5, q_6, q_7)
    return x, y, z, tr, ax1, ax2, ax3

def fk_franka_traj(q):
    q_1 = q[0:num]
    q_2 = q[num:2*num]
    q_3 = q[2*num:3*num]
    q_4 = q[3*num:4*num]
    q_5 = q[4*num:5*num]
    q_6 = q[5*num:6*num]
    q_7 = q[6*num:7*num]
    x, y, z, tr, ax1, ax2, ax3 = get_transforms(q_1, q_2, q_3, q_4, q_5, q_6, q_7)
    return x, y, z, tr, ax1, ax2, ax3

x_guess = data['traj'][0]['jointangles'].T.reshape(-1)
x_init, y_init, z_init, tr_init, ax1_init, ax2_init, ax3_init = fk_franka(np.hstack((q_1_init, q_2_init, q_3_init, q_4_init, q_5_init, q_6_init, q_7_init)))
x_fin, y_fin, z_fin, tr_fin, ax1_fin, ax2_fin, ax3_fin = fk_franka(np.hstack((q_1_fin, q_2_fin, q_3_fin, q_4_fin, q_5_fin, q_6_fin, q_7_fin)))

#### Cost Function
def cost_fun(q, params):
    x_fin, y_fin, z_fin = params
    
    q_1 = q[0:num]
    q_2 = q[num:2*num]
    q_3 = q[2*num:3*num]
    q_4 = q[3*num:4*num]
    q_5 = q[4*num:5*num]
    q_6 = q[5*num:6*num]
    q_7 = q[6*num:7*num]

    cq1 = np.cos(q_1)
    cq2 = np.cos(q_2)
    cq3 = np.cos(q_3)
    cq4 = np.cos(q_4)
    cq5 = np.cos(q_5)
    cq6 = np.cos(q_6)
    cq7 = np.cos(q_7)
    
    sq1 = np.sin(q_1)
    sq2 = np.sin(q_2)
    sq3 = np.sin(q_3)
    sq4 = np.sin(q_4)
    sq5 = np.sin(q_5)
    sq6 = np.sin(q_6)
    sq7 = np.sin(q_7)

    x = -0.107 * (((np.sin(q_1) * np.sin(q_3) - np.cos(q_1) * np.cos(q_2) * np.cos(q_3)) * np.cos(q_4) - np.sin(q_2) * np.sin(q_4) * np.cos(q_1)) * np.cos(q_5) + (np.sin(q_1) * np.cos(q_3) + np.sin(q_3) * np.cos(q_1) * np.cos(q_2)) * np.sin(q_5)) * np.sin(q_6) - 0.088 * (((np.sin(q_1) * np.sin(q_3) - np.cos(q_1) * np.cos(q_2) * np.cos(q_3)) * np.cos(q_4) - np.sin(q_2) * np.sin(q_4) * np.cos(q_1)) * np.cos(q_5) + (np.sin(q_1) * np.cos(q_3) + np.sin(q_3) * np.cos(q_1) * np.cos(q_2)) * np.sin(q_5)) * np.cos(q_6) + 0.088 * ((np.sin(q_1) * np.sin(q_3) - np.cos(q_1) * np.cos(q_2) * np.cos(q_3)) * np.sin(q_4) + np.sin(q_2) * np.cos(q_1) * np.cos(q_4)) * np.sin(q_6) - 0.107 * ((np.sin(q_1) * np.sin(q_3) - np.cos(q_1) * np.cos(q_2) * np.cos(q_3)) * np.sin(q_4) + np.sin(q_2) * np.cos(q_1) * np.cos(q_4)) * np.cos(q_6) + 0.384 * (np.sin(q_1) * np.sin(q_3) - np.cos(q_1) * np.cos(q_2) * np.cos(q_3)) * np.sin(q_4) + 0.0825 * (np.sin(q_1) * np.sin(q_3) - np.cos(q_1) * np.cos(q_2) * np.cos(q_3)) * np.cos(q_4) - 0.0825 * np.sin(q_1) * np.sin(q_3) - 0.0825 * np.sin(q_2) * np.sin(q_4) * np.cos(q_1) + 0.384 * np.sin(q_2) * np.cos(q_1) * np.cos(q_4) + 0.316 * np.sin(q_2) * np.cos(q_1) + 0.0825 * np.cos(q_1) * np.cos(q_2) * np.cos(q_3)
    y = 0.107 * (((np.sin(q_1) * np.cos(q_2) * np.cos(q_3) + np.sin(q_3) * np.cos(q_1)) * np.cos(q_4) + np.sin(q_1) * np.sin(q_2) * np.sin(q_4)) * np.cos(q_5) - (np.sin(q_1) * np.sin(q_3) * np.cos(q_2) - np.cos(q_1) * np.cos(q_3)) * np.sin(q_5)) * np.sin(q_6) + 0.088 * (((np.sin(q_1) * np.cos(q_2) * np.cos(q_3) + np.sin(q_3) * np.cos(q_1)) * np.cos(q_4) + np.sin(q_1) * np.sin(q_2) * np.sin(q_4)) * np.cos(q_5) - (np.sin(q_1) * np.sin(q_3) * np.cos(q_2) - np.cos(q_1) * np.cos(q_3)) * np.sin(q_5)) * np.cos(q_6) - 0.088 * ((np.sin(q_1) * np.cos(q_2) * np.cos(q_3) + np.sin(q_3) * np.cos(q_1)) * np.sin(q_4) - np.sin(q_1) * np.sin(q_2) * np.cos(q_4)) * np.sin(q_6) + 0.107 * ((np.sin(q_1) * np.cos(q_2) * np.cos(q_3) + np.sin(q_3) * np.cos(q_1)) * np.sin(q_4) - np.sin(q_1) * np.sin(q_2) * np.cos(q_4)) * np.cos(q_6) - 0.384 * (np.sin(q_1) * np.cos(q_2) * np.cos(q_3) + np.sin(q_3) * np.cos(q_1)) * np.sin(q_4) - 0.0825 * (np.sin(q_1) * np.cos(q_2) * np.cos(q_3) + np.sin(q_3) * np.cos(q_1)) * np.cos(q_4) - 0.0825 * np.sin(q_1) * np.sin(q_2) * np.sin(q_4) + 0.384 * np.sin(q_1) * np.sin(q_2) * np.cos(q_4) + 0.316 * np.sin(q_1) * np.sin(q_2) + 0.0825 * np.sin(q_1) * np.cos(q_2) * np.cos(q_3) + 0.0825 * np.sin(q_3) * np.cos(q_1)
    z = -0.107 * ((np.sin(q_2) * np.cos(q_3) * np.cos(q_4) - np.sin(q_4) * np.cos(q_2)) * np.cos(q_5) - np.sin(q_2) * np.sin(q_3) * np.sin(q_5)) * np.sin(q_6) - 0.088 * ((np.sin(q_2) * np.cos(q_3) * np.cos(q_4) - np.sin(q_4) * np.cos(q_2)) * np.cos(q_5) - np.sin(q_2) * np.sin(q_3) * np.sin(q_5)) * np.cos(q_6) + 0.088 * (np.sin(q_2) * np.sin(q_4) * np.cos(q_3) + np.cos(q_2) * np.cos(q_4)) * np.sin(q_6) - 0.107 * (np.sin(q_2) * np.sin(q_4) * np.cos(q_3) + np.cos(q_2) * np.cos(q_4)) * np.cos(q_6) + 0.384 * np.sin(q_2) * np.sin(q_4) * np.cos(q_3) + 0.0825 * np.sin(q_2) * np.cos(q_3) * np.cos(q_4) - 0.0825 * np.sin(q_2) * np.cos(q_3) - 0.0825 * np.sin(q_4) * np.cos(q_2) + 0.384 * np.cos(q_2) * np.cos(q_4) + 0.316 * np.cos(q_2) + 0.33
    
    r_32 = -cq7 * (cq5 * sq2 * sq3 - sq5 * (cq2 * sq4 - cq3 * cq4 * sq2)) - sq7 * (cq6 * (cq5 * (cq2 * sq4 - cq3 * cq4 * sq2) + sq2 * sq3 * sq5) + sq6 * (cq2 * cq4 + cq3 * sq2 * sq4))
    r_33 = -cq6 * (cq2 * cq4 + cq3 * sq2 * sq4) + sq6 * (cq5 * (cq2 * sq4 - cq3 * cq4 * sq2) + sq2 * sq3 * sq5)
    r_31 = cq7 * (cq6 * (cq5 * (cq2 * sq4 - cq3 * cq4 * sq2) + sq2 * sq3 * sq5) + sq6 * (cq2 * cq4 + cq3 * sq2 * sq4)) - sq7 * (cq5 * sq2 * sq3 - sq5 * (cq2 * sq4 - cq3 * cq4 * sq2))

    # Position cost
    f_pos_cost = (x[-1] - x_fin) ** 2 + (y[-1] - y_fin) ** 2 + (z[-1] - z_fin) ** 2 + (x[0] - x_init) ** 2 + (y[0] - y_init) ** 2 + (z[0] - z_init) ** 2
    # Orientation cost
    f_orient_cost = np.sum((np.arctan2(r_32, r_33) - roll_des) ** 2 + (-np.arcsin(r_31) - pitch_des) ** 2)
    
    # Smoothness cost
    f_smoothness_vel = np.sum(np.dot(A_vel, q_1) ** 2) + np.sum(np.dot(A_vel, q_2) ** 2) + np.sum(np.dot(A_vel, q_3) ** 2) + np.sum(np.dot(A_vel, q_4) ** 2) + np.sum(np.dot(A_vel, q_5) ** 2) + np.sum(np.dot(A_vel, q_6) ** 2) + np.sum(np.dot(A_vel, q_7) ** 2)
    f_smoothness_acc = np.sum(np.dot(A_acc, q_1) ** 2) + np.sum(np.dot(A_acc, q_2) ** 2) + np.sum(np.dot(A_acc, q_3) ** 2) + np.sum(np.dot(A_acc, q_4) ** 2) + np.sum(np.dot(A_acc, q_5) ** 2) + np.sum(np.dot(A_acc, q_6) ** 2) + np.sum(np.dot(A_acc, q_7) ** 2)
    f_smoothness_jerk = np.sum(np.dot(A_jerk, q_1) ** 2) + np.sum(np.dot(A_jerk, q_2) ** 2) + np.sum(np.dot(A_jerk, q_3) ** 2) + np.sum(np.dot(A_jerk, q_4) ** 2) + np.sum(np.dot(A_jerk, q_5) ** 2) + np.sum(np.dot(A_jerk, q_6) ** 2) + np.sum(np.dot(A_jerk, q_7) ** 2)
    f_smoothness_cost = rho_vel * f_smoothness_vel + rho_acc * f_smoothness_acc + rho_jerk * f_smoothness_jerk

    cost = weight_orient * f_orient_cost + f_smoothness_cost + weight_pos * f_pos_cost
    return cost

#### Compute solution
print("Computing initial solution")
opts = {
    'maxiter': 600
}

bnds = Bounds(q_min_traj, q_max_traj)
cost_grad = grad(cost_fun)
init_params = np.array([x_fin, y_fin, z_fin])

start = time.time()
start_solver_process_time = time.process_time()
res = minimize(cost_fun, x_guess, init_params, method= 'SLSQP', jac=cost_grad, bounds=bnds, options=opts)
end_solver_process_time = time.process_time()
end_time = time.time()
print(end_time - start, res.message)

initial_cost1 = cost_fun(res.x, init_params)
print("Initial cost: {}".format(initial_cost1))

#### Generate argmin solutions and trajectories
num_samples = 6
flag = 1
cur_traj_cnt = 1

numpy_dict = {
    "initial_soln": res.x
}

numpy_time_dict = {
    "initial_soln": {
        "cpu_time": end_time - start,
        "process_time": end_solver_process_time - start_solver_process_time
    }
}

for num_iter in range(1, num_samples + 1):
    for sample_num in range(len(data['q_fin'][num_iter])):
        print("ITERATION {}, Cur traj num: {}".format(num_iter, cur_traj_cnt))
        print("Saving solution")

        q_fin_new = data['q_fin'][num_iter][sample_num]
        q_1_fin_new, q_2_fin_new, q_3_fin_new, q_4_fin_new, q_5_fin_new, q_6_fin_new, q_7_fin_new = q_fin_new
        x_fin_new, y_fin_new, z_fin_new, _, _, _, _ = fk_franka(np.hstack((q_1_fin_new, q_2_fin_new, q_3_fin_new, q_4_fin_new, q_5_fin_new, q_6_fin_new, q_7_fin_new)))

        perturbed_params = np.array([x_fin_new, y_fin_new, z_fin_new])

        start_solver_time = time.time()
        start_solver_process_time = time.process_time()
        res_new = minimize(cost_fun, res.x, perturbed_params, method= 'SLSQP', jac=cost_grad, bounds=bnds, options=opts)
        end_solver_process_time = time.process_time()
        end_solver_time = time.time()
        print("Solver time taken: {}".format(end_solver_time - start_solver_time))
        print("Solver process time: {}".format(end_solver_process_time - start_solver_process_time))
        print("Solver message: {}".format(res_new.message))
        print("-"*100)

        numpy_dict[cur_traj_cnt] = res_new.x
        numpy_time_dict[cur_traj_cnt] = {
            "cpu_time": end_solver_time - start_solver_time,
            "process_time": end_solver_process_time - start_solver_process_time
        }
        cur_traj_cnt += 1

os.makedirs(args.results_path, exist_ok=True)
np_path = os.path.join(args.results_path, 'traj_{}.npy'.format(cur_pos + 1))
np.save(np_path, numpy_dict)

np_time_path = os.path.join(args.results_path, 'traj_time_{}.npy'.format(cur_pos + 1))
np.save(np_time_path, numpy_time_dict)
