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
parser.add_argument("--soln_path", help="Solution path", type=str)
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

cur_soln_path = os.path.join(args.soln_path, 'traj_{}.npy'.format(cur_pos + 1))
soln = np.load(cur_soln_path, allow_pickle=True)
soln = soln.item()

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

def get_cost_components(q, params):
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

    return f_pos_cost, f_orient_cost, f_smoothness_vel, f_smoothness_acc, f_smoothness_jerk, np.arctan2(r_32, r_33), -np.arcsin(r_31)

#### Compute solution
print("Computing initial solution")
opts = {
    'maxiter': 1
}

bnds = Bounds(q_min_traj, q_max_traj)
cost_grad = grad(cost_fun)

init_params = np.array([x_fin, y_fin, z_fin])

start = time.time()
res = minimize(cost_fun, x_guess, init_params, method= 'SLSQP', jac=cost_grad, bounds=bnds, options=opts)
end_time = time.time()

res.x = soln['initial_soln']
initial_cost1 = cost_fun(res.x, init_params)

#### Plot trajectory
def get_trajectory(sol):
    q_1 = sol[0:num]
    q_2 = sol[num:2*num]
    q_3 = sol[2*num:3*num]
    q_4 = sol[3*num:4*num]
    q_5 = sol[4*num:5*num]
    q_6 = sol[5*num:6*num]
    q_7 = sol[6*num:7*num]
    x, y, z, tr, ax1, ax2, ax3 = fk_franka_traj(sol)
    return q_1, q_2, q_3, q_4, q_5, q_6, q_7, x, y, z, tr, ax1, ax2, ax3

q_1, q_2, q_3, q_4, q_5, q_6, q_7, x, y, z, tr, ax1, ax2, ax3 = get_trajectory(res.x)

#### Plotting code
def save_trajectory_plots(num_iter, x_init, y_init, z_init, x_fin, y_fin, z_fin, x, y, z, message):
    fig = plt.figure(figsize=(10, 8))
    axs = fig.add_subplot(111, projection='3d')
    axs.plot(x_init * np.ones(1), y_init * np.ones(1), z_init * np.ones(1), 'om', markersize=10)
    axs.plot(x_fin * np.ones(1), y_fin * np.ones(1), z_fin * np.ones(1), 'og', markersize=10)
    axs.plot(x, y, z, '-o', linewidth=3.0, markersize=6.0)
    axs.set_xlim3d(-1.0, 1.0)
    axs.set_ylim3d(-1.0, 1.0)
    axs.set_zlim3d(-0.3, 1.2)

    fig.suptitle('trajectory_{}'.format(message))
    plot_dir = os.path.join(args.results_path, 'traj_{}'.format(cur_pos + 1), 'sample_{}'.format(num_iter))
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, 'trajectory_{}.png'.format(message))
    fig.savefig(plot_path)
    plt.close(fig)

def save_q_vals_plots(num_iter, q_1, q_2, q_3, q_4, q_5, q_6, q_7, message):
    fig, axs = plt.subplots(2, 4, figsize=(16, 6))
    axs[0, 0].plot(q_1, '-o', linewidth = 3.0, markersize = 4.0)
    axs[0, 1].plot(q_2, '-o', linewidth = 3.0, markersize = 4.0)
    axs[0, 2].plot(q_3, '-o', linewidth = 3.0, markersize = 4.0)
    axs[0, 3].plot(q_4, '-o', linewidth = 3.0, markersize = 4.0)
    axs[1, 0].plot(q_5, '-o', linewidth = 3.0, markersize = 4.0)
    axs[1, 1].plot(q_6, '-o', linewidth = 3.0, markersize = 4.0)
    axs[1, 2].plot(q_7, '-o', linewidth = 3.0, markersize = 4.0)

    fig.suptitle('q_vals_{}'.format(message))
    plot_dir = os.path.join(args.results_path, 'traj_{}'.format(cur_pos + 1), 'sample_{}'.format(num_iter))
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, 'q_vals_{}.png'.format(message))
    fig.savefig(plot_path)
    plt.close(fig)

#### Trajectory plotting code
def save_trajectory_plots_comparison(num_iter, x_init, y_init, z_init, x_fin, y_fin, z_fin, x, y, z, x_fin_new, y_fin_new, \
    z_fin_new, x_true, y_true, z_true, x_pred, y_pred, z_pred, total_dis, cost_dict, best_cost, message):

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Old Traj
    ax.plot(x_init * np.ones(1), y_init * np.ones(1), z_init * np.ones(1), 'om', markersize=10, label='Initial point')
    ax.plot(x_fin * np.ones(1), y_fin * np.ones(1), z_fin * np.ones(1), 'ob', markersize=10, label='Final point')
    ax.plot(x, y, z, '-o', linewidth=3.0, markersize=3.0, alpha=0.5, label='Old Traj')

    # Predicted Traj
    ax.plot(x_pred, y_pred, z_pred, '-o', linewidth=3.0, markersize=7.0, label='Predicted Traj')

    # Actual Traj
    ax.plot(x_true, y_true, z_true, '-o', linewidth=3.0, markersize=4.0, alpha=0.4, label='Actual Traj')
    ax.plot(x_fin_new * np.ones(1), y_fin_new * np.ones(1), z_fin_new * np.ones(1), 'og', markersize=10, label='Perturbed Point')

    # ax.legend()
    ax.set_xlim3d(-1.0, 1.0)
    ax.set_ylim3d(-1.0, 1.0)
    ax.set_zlim3d(-0.3, 1.2)

    patch1 = mpatches.Patch(color='black', label="True Cost: {}".format(cost_dict["actual_cost"]))
    patch2 = mpatches.Patch(color='black', label="Best Cost: {}".format(best_cost))
    patch3 = mpatches.Patch(color='black', label="Cost old pos: {}".format(cost_dict["initial_cost_old_pos"]))
    patch4 = mpatches.Patch(color='black', label="Cost new pos: {}".format(cost_dict["initial_cost_new_pos"]))
    patch5 = mpatches.Patch(color='black', label="Total Perturbation: {}".format(cost_dict["total_perturbation"]))
    patch6 = mpatches.Patch(color='m', label="Initial point")
    patch7 = mpatches.Patch(color='b', label='Final point')
    patch8 = mpatches.Patch(color='g', label='Perturbed point')

    ax.legend(handles=[patch1, patch2, patch3, patch4, patch5, patch6, patch7, patch8], loc=4)

    fig.suptitle('trajectory_{}, total perturbation norm: {}'.format(message, total_dis))
    plot_dir = os.path.join(args.results_path, 'traj_{}'.format(cur_pos + 1), 'sample_{}'.format(num_iter))
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, 'trajectory_{}.png'.format(message))
    fig.savefig(plot_path)
    plt.close(fig)

def save_q_vals_plots_compare(num_iter, q_1_true, q_2_true, q_3_true, q_4_true, q_5_true, q_6_true, q_7_true, q_1_pred, \
    q_2_pred, q_3_pred, q_4_pred, q_5_pred, q_6_pred, q_7_pred, total_dis, cost_dict, message):
    fig, axs = plt.subplots(2, 4, figsize=(16, 6))

    axs[0, 0].plot(q_1_true, '-o', linewidth = 3.0, markersize = 4.0, alpha=0.5, label='True')
    axs[0, 0].plot(q_1_pred, linewidth = 3.0, markersize = 6.0, label='Predicted')
    axs[0, 0].legend()

    axs[0, 1].plot(q_2_true, '-o', linewidth = 3.0, markersize = 4.0, alpha=0.5, label='True')
    axs[0, 1].plot(q_2_pred, linewidth = 3.0, markersize = 6.0, label='Predicted')
    axs[0, 1].legend()

    axs[0, 2].plot(q_3_true, '-o', linewidth = 3.0, markersize = 4.0, alpha=0.5, label='True')
    axs[0, 2].plot(q_3_pred, linewidth = 3.0, markersize = 6.0, label='Predicted')
    axs[0, 2].legend()

    axs[0, 3].plot(q_4_true, '-o', linewidth = 3.0, markersize = 4.0, alpha=0.5, label='True')
    axs[0, 3].plot(q_4_pred, linewidth = 3.0, markersize = 6.0, label='Predicted')
    axs[0, 3].legend()

    axs[1, 0].plot(q_5_true, '-o', linewidth = 3.0, markersize = 4.0, alpha=0.5, label='True')
    axs[1, 0].plot(q_5_pred, linewidth = 3.0, markersize = 6.0, label='Predicted')
    axs[1, 0].legend()

    axs[1, 1].plot(q_6_true, '-o', linewidth = 3.0, markersize = 4.0, alpha=0.5, label='True')
    axs[1, 1].plot(q_6_pred, linewidth = 3.0, markersize = 6.0, label='Predicted')
    axs[1, 1].legend()

    axs[1, 2].plot(q_7_true, '-o', linewidth = 3.0, markersize = 4.0, alpha=0.5, label='True')
    axs[1, 2].plot(q_7_pred, linewidth = 3.0, markersize = 6.0, label='Predicted')
    axs[1, 2].legend()

    fig.suptitle('q_vals_{}, total perturbation norm: {}'.format(message, total_dis))
    plot_dir = os.path.join(args.results_path, 'traj_{}'.format(cur_pos + 1), 'sample_{}'.format(num_iter))
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, 'q_vals_{}.png'.format(message))
    fig.savefig(plot_path)
    plt.close(fig)

def save_roll_plots(num_iter, roll_pred, roll_actual, message):
    fig, axs = plt.subplots(1, 1, figsize=(8, 6))

    axs.plot(roll_actual, '-o', linewidth = 3.0, markersize = 4.0, alpha=0.5, label='True')
    axs.plot(roll_pred, linewidth = 3.0, markersize = 6.0, label='Predicted')
    axs.set_ylim(-2.0, 0.0)
    axs.legend()

    fig.suptitle('roll_comparison_plot')
    plot_dir = os.path.join(args.results_path, 'traj_{}'.format(cur_pos + 1), 'sample_{}'.format(num_iter))
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, 'roll_{}.png'.format(message))
    fig.savefig(plot_path)
    plt.close(fig)

def save_pitch_plots(num_iter, pitch_pred, pitch_actual, message):
    fig, axs = plt.subplots(1, 1, figsize=(8, 6))

    axs.plot(pitch_actual, '-o', linewidth = 3.0, markersize = 4.0, alpha=0.5, label='True')
    axs.plot(pitch_pred, linewidth = 3.0, markersize = 6.0, label='Predicted')
    axs.set_ylim(-2.0, 0.5)
    axs.legend()

    fig.suptitle('pitch_comparison_plot')
    plot_dir = os.path.join(args.results_path, 'traj_{}'.format(cur_pos + 1), 'sample_{}'.format(num_iter))
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, 'pitch_{}.png'.format(message))
    fig.savefig(plot_path)
    plt.close(fig)

#### Compute Perturbed solution
def get_transforms_jax(q_1, q_2, q_3, q_4, q_5, q_6, q_7):
    x = -0.107 * (((jnp.sin(q_1) * jnp.sin(q_3) - jnp.cos(q_1) * jnp.cos(q_2) * jnp.cos(q_3)) * jnp.cos(q_4) - jnp.sin(q_2) * jnp.sin(q_4) * jnp.cos(q_1)) * jnp.cos(q_5) + (jnp.sin(q_1) * jnp.cos(q_3) + jnp.sin(q_3) * jnp.cos(q_1) * jnp.cos(q_2)) * jnp.sin(q_5)) * jnp.sin(q_6) - 0.088 * (((jnp.sin(q_1) * jnp.sin(q_3) - jnp.cos(q_1) * jnp.cos(q_2) * jnp.cos(q_3)) * jnp.cos(q_4) - jnp.sin(q_2) * jnp.sin(q_4) * jnp.cos(q_1)) * jnp.cos(q_5) + (jnp.sin(q_1) * jnp.cos(q_3) + jnp.sin(q_3) * jnp.cos(q_1) * jnp.cos(q_2)) * jnp.sin(q_5)) * jnp.cos(q_6) + 0.088 * ((jnp.sin(q_1) * jnp.sin(q_3) - jnp.cos(q_1) * jnp.cos(q_2) * jnp.cos(q_3)) * jnp.sin(q_4) + jnp.sin(q_2) * jnp.cos(q_1) * jnp.cos(q_4)) * jnp.sin(q_6) - 0.107 * ((jnp.sin(q_1) * jnp.sin(q_3) - jnp.cos(q_1) * jnp.cos(q_2) * jnp.cos(q_3)) * jnp.sin(q_4) + jnp.sin(q_2) * jnp.cos(q_1) * jnp.cos(q_4)) * jnp.cos(q_6) + 0.384 * (jnp.sin(q_1) * jnp.sin(q_3) - jnp.cos(q_1) * jnp.cos(q_2) * jnp.cos(q_3)) * jnp.sin(q_4) + 0.0825 * (jnp.sin(q_1) * jnp.sin(q_3) - jnp.cos(q_1) * jnp.cos(q_2) * jnp.cos(q_3)) * jnp.cos(q_4) - 0.0825 * jnp.sin(q_1) * jnp.sin(q_3) - 0.0825 * jnp.sin(q_2) * jnp.sin(q_4) * jnp.cos(q_1) + 0.384 * jnp.sin(q_2) * jnp.cos(q_1) * jnp.cos(q_4) + 0.316 * jnp.sin(q_2) * jnp.cos(q_1) + 0.0825 * jnp.cos(q_1) * jnp.cos(q_2) * jnp.cos(q_3)
    y = 0.107 * (((jnp.sin(q_1) * jnp.cos(q_2) * jnp.cos(q_3) + jnp.sin(q_3) * jnp.cos(q_1)) * jnp.cos(q_4) + jnp.sin(q_1) * jnp.sin(q_2) * jnp.sin(q_4)) * jnp.cos(q_5) - (jnp.sin(q_1) * jnp.sin(q_3) * jnp.cos(q_2) - jnp.cos(q_1) * jnp.cos(q_3)) * jnp.sin(q_5)) * jnp.sin(q_6) + 0.088 * (((jnp.sin(q_1) * jnp.cos(q_2) * jnp.cos(q_3) + jnp.sin(q_3) * jnp.cos(q_1)) * jnp.cos(q_4) + jnp.sin(q_1) * jnp.sin(q_2) * jnp.sin(q_4)) * jnp.cos(q_5) - (jnp.sin(q_1) * jnp.sin(q_3) * jnp.cos(q_2) - jnp.cos(q_1) * jnp.cos(q_3)) * jnp.sin(q_5)) * jnp.cos(q_6) - 0.088 * ((jnp.sin(q_1) * jnp.cos(q_2) * jnp.cos(q_3) + jnp.sin(q_3) * jnp.cos(q_1)) * jnp.sin(q_4) - jnp.sin(q_1) * jnp.sin(q_2) * jnp.cos(q_4)) * jnp.sin(q_6) + 0.107 * ((jnp.sin(q_1) * jnp.cos(q_2) * jnp.cos(q_3) + jnp.sin(q_3) * jnp.cos(q_1)) * jnp.sin(q_4) - jnp.sin(q_1) * jnp.sin(q_2) * jnp.cos(q_4)) * jnp.cos(q_6) - 0.384 * (jnp.sin(q_1) * jnp.cos(q_2) * jnp.cos(q_3) + jnp.sin(q_3) * jnp.cos(q_1)) * jnp.sin(q_4) - 0.0825 * (jnp.sin(q_1) * jnp.cos(q_2) * jnp.cos(q_3) + jnp.sin(q_3) * jnp.cos(q_1)) * jnp.cos(q_4) - 0.0825 * jnp.sin(q_1) * jnp.sin(q_2) * jnp.sin(q_4) + 0.384 * jnp.sin(q_1) * jnp.sin(q_2) * jnp.cos(q_4) + 0.316 * jnp.sin(q_1) * jnp.sin(q_2) + 0.0825 * jnp.sin(q_1) * jnp.cos(q_2) * jnp.cos(q_3) + 0.0825 * jnp.sin(q_3) * jnp.cos(q_1)
    z = -0.107 * ((jnp.sin(q_2) * jnp.cos(q_3) * jnp.cos(q_4) - jnp.sin(q_4) * jnp.cos(q_2)) * jnp.cos(q_5) - jnp.sin(q_2) * jnp.sin(q_3) * jnp.sin(q_5)) * jnp.sin(q_6) - 0.088 * ((jnp.sin(q_2) * jnp.cos(q_3) * jnp.cos(q_4) - jnp.sin(q_4) * jnp.cos(q_2)) * jnp.cos(q_5) - jnp.sin(q_2) * jnp.sin(q_3) * jnp.sin(q_5)) * jnp.cos(q_6) + 0.088 * (jnp.sin(q_2) * jnp.sin(q_4) * jnp.cos(q_3) + jnp.cos(q_2) * jnp.cos(q_4)) * jnp.sin(q_6) - 0.107 * (jnp.sin(q_2) * jnp.sin(q_4) * jnp.cos(q_3) + jnp.cos(q_2) * jnp.cos(q_4)) * jnp.cos(q_6) + 0.384 * jnp.sin(q_2) * jnp.sin(q_4) * jnp.cos(q_3) + 0.0825 * jnp.sin(q_2) * jnp.cos(q_3) * jnp.cos(q_4) - 0.0825 * jnp.sin(q_2) * jnp.cos(q_3) - 0.0825 * jnp.sin(q_4) * jnp.cos(q_2) + 0.384 * jnp.cos(q_2) * jnp.cos(q_4) + 0.316 * jnp.cos(q_2) + 0.33
    tr = jnp.sin(q_2) * jnp.sin(q_3) * jnp.sin(q_5) * jnp.sin(q_6) + jnp.sin(q_2) * jnp.sin(q_4) * jnp.sin(q_5) * jnp.sin(q_1 + q_7) - jnp.sin(q_2) * jnp.sin(q_4) * jnp.cos(q_3) * jnp.cos(q_6) + jnp.sin(q_2) * jnp.sin(q_4) * jnp.cos(q_5) * jnp.cos(q_6) * jnp.cos(q_1 + q_7) - jnp.sin(q_2) * jnp.sin(q_6) * jnp.cos(q_3) * jnp.cos(q_4) * jnp.cos(q_5) + jnp.sin(q_2) * jnp.sin(q_6) * jnp.cos(q_4) * jnp.cos(q_1 + q_7) + jnp.sin(q_3) * jnp.sin(q_4) * jnp.sin(q_6) * jnp.sin(q_1 + q_7) - jnp.sin(q_3) * jnp.sin(q_5) * jnp.cos(q_2) * jnp.cos(q_6) * jnp.cos(q_1 + q_7) + jnp.sin(q_3) * jnp.sin(q_5) * jnp.cos(q_4) * jnp.cos(q_1 + q_7) + jnp.sin(q_3) * jnp.sin(q_1 + q_7) * jnp.cos(q_2) * jnp.cos(q_5) - jnp.sin(q_3) * jnp.sin(q_1 + q_7) * jnp.cos(q_4) * jnp.cos(q_5) * jnp.cos(q_6) - jnp.sin(q_4) * jnp.sin(q_6) * jnp.cos(q_2) * jnp.cos(q_3) * jnp.cos(q_1 + q_7) + jnp.sin(q_4) * jnp.sin(q_6) * jnp.cos(q_2) * jnp.cos(q_5) + jnp.sin(q_5) * jnp.sin(q_1 + q_7) * jnp.cos(q_2) * jnp.cos(q_3) * jnp.cos(q_4) - jnp.sin(q_5) * jnp.sin(q_1 + q_7) * jnp.cos(q_3) * jnp.cos(q_6) + jnp.cos(q_2) * jnp.cos(q_3) * jnp.cos(q_4) * jnp.cos(q_5) * jnp.cos(q_6) * jnp.cos(q_1 + q_7) - jnp.cos(q_2) * jnp.cos(q_4) * jnp.cos(q_6) - jnp.cos(q_3) * jnp.cos(q_5) * jnp.cos(q_1 + q_7)
    ax1 = -0.5 * (((jnp.sin(q_1) * jnp.cos(q_2) * jnp.cos(q_3) + jnp.sin(q_3) * jnp.cos(q_1)) * jnp.cos(q_4) + jnp.sin(q_1) * jnp.sin(q_2) * jnp.sin(q_4)) * jnp.cos(q_5) - (jnp.sin(q_1) * jnp.sin(q_3) * jnp.cos(q_2) - jnp.cos(q_1) * jnp.cos(q_3)) * jnp.sin(q_5)) * jnp.sin(q_6) + 0.5 * (((jnp.sin(q_2) * jnp.cos(q_3) * jnp.cos(q_4) - jnp.sin(q_4) * jnp.cos(q_2)) * jnp.cos(q_5) - jnp.sin(q_2) * jnp.sin(q_3) * jnp.sin(q_5)) * jnp.cos(q_6) - (jnp.sin(q_2) * jnp.sin(q_4) * jnp.cos(q_3) + jnp.cos(q_2) * jnp.cos(q_4)) * jnp.sin(q_6)) * jnp.sin(q_7) - 0.5 * ((jnp.sin(q_1) * jnp.cos(q_2) * jnp.cos(q_3) + jnp.sin(q_3) * jnp.cos(q_1)) * jnp.sin(q_4) - jnp.sin(q_1) * jnp.sin(q_2) * jnp.cos(q_4)) * jnp.cos(q_6) - 0.5 * ((jnp.sin(q_2) * jnp.cos(q_3) * jnp.cos(q_4) - jnp.sin(q_4) * jnp.cos(q_2)) * jnp.sin(q_5) + jnp.sin(q_2) * jnp.sin(q_3) * jnp.cos(q_5)) * jnp.cos(q_7)
    ax2 = -0.5 * (((jnp.sin(q_1) * jnp.sin(q_3) - jnp.cos(q_1) * jnp.cos(q_2) * jnp.cos(q_3)) * jnp.cos(q_4) - jnp.sin(q_2) * jnp.sin(q_4) * jnp.cos(q_1)) * jnp.cos(q_5) + (jnp.sin(q_1) * jnp.cos(q_3) + jnp.sin(q_3) * jnp.cos(q_1) * jnp.cos(q_2)) * jnp.sin(q_5)) * jnp.sin(q_6) + 0.5 * (((jnp.sin(q_2) * jnp.cos(q_3) * jnp.cos(q_4) - jnp.sin(q_4) * jnp.cos(q_2)) * jnp.cos(q_5) - jnp.sin(q_2) * jnp.sin(q_3) * jnp.sin(q_5)) * jnp.cos(q_6) - (jnp.sin(q_2) * jnp.sin(q_4) * jnp.cos(q_3) + jnp.cos(q_2) * jnp.cos(q_4)) * jnp.sin(q_6)) * jnp.cos(q_7) - 0.5 * ((jnp.sin(q_1) * jnp.sin(q_3) - jnp.cos(q_1) * jnp.cos(q_2) * jnp.cos(q_3)) * jnp.sin(q_4) + jnp.sin(q_2) * jnp.cos(q_1) * jnp.cos(q_4)) * jnp.cos(q_6) + 0.5 * ((jnp.sin(q_2) * jnp.cos(q_3) * jnp.cos(q_4) - jnp.sin(q_4) * jnp.cos(q_2)) * jnp.sin(q_5) + jnp.sin(q_2) * jnp.sin(q_3) * jnp.cos(q_5)) * jnp.sin(q_7)
    ax3 = -0.5 * jnp.sin(q_2) * jnp.sin(q_4) * jnp.sin(q_5) * jnp.cos(q_1 + q_7) + 0.5 * jnp.sin(q_2) * jnp.sin(q_4) * jnp.sin(q_1 + q_7) * jnp.cos(q_5) * jnp.cos(q_6) + 0.5 * jnp.sin(q_2) * jnp.sin(q_6) * jnp.sin(q_1 + q_7) * jnp.cos(q_4) - 0.5 * jnp.sin(q_3) * jnp.sin(q_4) * jnp.sin(q_6) * jnp.cos(q_1 + q_7) - 0.5 * jnp.sin(q_3) * jnp.sin(q_5) * jnp.sin(q_1 + q_7) * jnp.cos(q_2) * jnp.cos(q_6) + 0.5 * jnp.sin(q_3) * jnp.sin(q_5) * jnp.sin(q_1 + q_7) * jnp.cos(q_4) - 0.5 * jnp.sin(q_3) * jnp.cos(q_2) * jnp.cos(q_5) * jnp.cos(q_1 + q_7) + 0.5 * jnp.sin(q_3) * jnp.cos(q_4) * jnp.cos(q_5) * jnp.cos(q_6) * jnp.cos(q_1 + q_7) - 0.5 * jnp.sin(q_4) * jnp.sin(q_6) * jnp.sin(q_1 + q_7) * jnp.cos(q_2) * jnp.cos(q_3) - 0.5 * jnp.sin(q_5) * jnp.cos(q_2) * jnp.cos(q_3) * jnp.cos(q_4) * jnp.cos(q_1 + q_7) + 0.5 * jnp.sin(q_5) * jnp.cos(q_3) * jnp.cos(q_6) * jnp.cos(q_1 + q_7) + 0.5 * jnp.sin(q_1 + q_7) * jnp.cos(q_2) * jnp.cos(q_3) * jnp.cos(q_4) * jnp.cos(q_5) * jnp.cos(q_6) - 0.5 * jnp.sin(q_1 + q_7) * jnp.cos(q_3) * jnp.cos(q_5)
    return x, y, z, tr, ax1, ax2, ax3

def cost_fun_jax(q, params):
    x_fin, y_fin, z_fin = params
    
    q_1 = q[0:num]
    q_2 = q[num:2*num]
    q_3 = q[2*num:3*num]
    q_4 = q[3*num:4*num]
    q_5 = q[4*num:5*num]
    q_6 = q[5*num:6*num]
    q_7 = q[6*num:7*num]

    cq1 = jnp.cos(q_1)
    cq2 = jnp.cos(q_2)
    cq3 = jnp.cos(q_3)
    cq4 = jnp.cos(q_4)
    cq5 = jnp.cos(q_5)
    cq6 = jnp.cos(q_6)
    cq7 = jnp.cos(q_7)
    
    sq1 = jnp.sin(q_1)
    sq2 = jnp.sin(q_2)
    sq3 = jnp.sin(q_3)
    sq4 = jnp.sin(q_4)
    sq5 = jnp.sin(q_5)
    sq6 = jnp.sin(q_6)
    sq7 = jnp.sin(q_7)

    x = -0.107 * (((jnp.sin(q_1) * jnp.sin(q_3) - jnp.cos(q_1) * jnp.cos(q_2) * jnp.cos(q_3)) * jnp.cos(q_4) - jnp.sin(q_2) * jnp.sin(q_4) * jnp.cos(q_1)) * jnp.cos(q_5) + (jnp.sin(q_1) * jnp.cos(q_3) + jnp.sin(q_3) * jnp.cos(q_1) * jnp.cos(q_2)) * jnp.sin(q_5)) * jnp.sin(q_6) - 0.088 * (((jnp.sin(q_1) * jnp.sin(q_3) - jnp.cos(q_1) * jnp.cos(q_2) * jnp.cos(q_3)) * jnp.cos(q_4) - jnp.sin(q_2) * jnp.sin(q_4) * jnp.cos(q_1)) * jnp.cos(q_5) + (jnp.sin(q_1) * jnp.cos(q_3) + jnp.sin(q_3) * jnp.cos(q_1) * jnp.cos(q_2)) * jnp.sin(q_5)) * jnp.cos(q_6) + 0.088 * ((jnp.sin(q_1) * jnp.sin(q_3) - jnp.cos(q_1) * jnp.cos(q_2) * jnp.cos(q_3)) * jnp.sin(q_4) + jnp.sin(q_2) * jnp.cos(q_1) * jnp.cos(q_4)) * jnp.sin(q_6) - 0.107 * ((jnp.sin(q_1) * jnp.sin(q_3) - jnp.cos(q_1) * jnp.cos(q_2) * jnp.cos(q_3)) * jnp.sin(q_4) + jnp.sin(q_2) * jnp.cos(q_1) * jnp.cos(q_4)) * jnp.cos(q_6) + 0.384 * (jnp.sin(q_1) * jnp.sin(q_3) - jnp.cos(q_1) * jnp.cos(q_2) * jnp.cos(q_3)) * jnp.sin(q_4) + 0.0825 * (jnp.sin(q_1) * jnp.sin(q_3) - jnp.cos(q_1) * jnp.cos(q_2) * jnp.cos(q_3)) * jnp.cos(q_4) - 0.0825 * jnp.sin(q_1) * jnp.sin(q_3) - 0.0825 * jnp.sin(q_2) * jnp.sin(q_4) * jnp.cos(q_1) + 0.384 * jnp.sin(q_2) * jnp.cos(q_1) * jnp.cos(q_4) + 0.316 * jnp.sin(q_2) * jnp.cos(q_1) + 0.0825 * jnp.cos(q_1) * jnp.cos(q_2) * jnp.cos(q_3)
    y = 0.107 * (((jnp.sin(q_1) * jnp.cos(q_2) * jnp.cos(q_3) + jnp.sin(q_3) * jnp.cos(q_1)) * jnp.cos(q_4) + jnp.sin(q_1) * jnp.sin(q_2) * jnp.sin(q_4)) * jnp.cos(q_5) - (jnp.sin(q_1) * jnp.sin(q_3) * jnp.cos(q_2) - jnp.cos(q_1) * jnp.cos(q_3)) * jnp.sin(q_5)) * jnp.sin(q_6) + 0.088 * (((jnp.sin(q_1) * jnp.cos(q_2) * jnp.cos(q_3) + jnp.sin(q_3) * jnp.cos(q_1)) * jnp.cos(q_4) + jnp.sin(q_1) * jnp.sin(q_2) * jnp.sin(q_4)) * jnp.cos(q_5) - (jnp.sin(q_1) * jnp.sin(q_3) * jnp.cos(q_2) - jnp.cos(q_1) * jnp.cos(q_3)) * jnp.sin(q_5)) * jnp.cos(q_6) - 0.088 * ((jnp.sin(q_1) * jnp.cos(q_2) * jnp.cos(q_3) + jnp.sin(q_3) * jnp.cos(q_1)) * jnp.sin(q_4) - jnp.sin(q_1) * jnp.sin(q_2) * jnp.cos(q_4)) * jnp.sin(q_6) + 0.107 * ((jnp.sin(q_1) * jnp.cos(q_2) * jnp.cos(q_3) + jnp.sin(q_3) * jnp.cos(q_1)) * jnp.sin(q_4) - jnp.sin(q_1) * jnp.sin(q_2) * jnp.cos(q_4)) * jnp.cos(q_6) - 0.384 * (jnp.sin(q_1) * jnp.cos(q_2) * jnp.cos(q_3) + jnp.sin(q_3) * jnp.cos(q_1)) * jnp.sin(q_4) - 0.0825 * (jnp.sin(q_1) * jnp.cos(q_2) * jnp.cos(q_3) + jnp.sin(q_3) * jnp.cos(q_1)) * jnp.cos(q_4) - 0.0825 * jnp.sin(q_1) * jnp.sin(q_2) * jnp.sin(q_4) + 0.384 * jnp.sin(q_1) * jnp.sin(q_2) * jnp.cos(q_4) + 0.316 * jnp.sin(q_1) * jnp.sin(q_2) + 0.0825 * jnp.sin(q_1) * jnp.cos(q_2) * jnp.cos(q_3) + 0.0825 * jnp.sin(q_3) * jnp.cos(q_1)
    z = -0.107 * ((jnp.sin(q_2) * jnp.cos(q_3) * jnp.cos(q_4) - jnp.sin(q_4) * jnp.cos(q_2)) * jnp.cos(q_5) - jnp.sin(q_2) * jnp.sin(q_3) * jnp.sin(q_5)) * jnp.sin(q_6) - 0.088 * ((jnp.sin(q_2) * jnp.cos(q_3) * jnp.cos(q_4) - jnp.sin(q_4) * jnp.cos(q_2)) * jnp.cos(q_5) - jnp.sin(q_2) * jnp.sin(q_3) * jnp.sin(q_5)) * jnp.cos(q_6) + 0.088 * (jnp.sin(q_2) * jnp.sin(q_4) * jnp.cos(q_3) + jnp.cos(q_2) * jnp.cos(q_4)) * jnp.sin(q_6) - 0.107 * (jnp.sin(q_2) * jnp.sin(q_4) * jnp.cos(q_3) + jnp.cos(q_2) * jnp.cos(q_4)) * jnp.cos(q_6) + 0.384 * jnp.sin(q_2) * jnp.sin(q_4) * jnp.cos(q_3) + 0.0825 * jnp.sin(q_2) * jnp.cos(q_3) * jnp.cos(q_4) - 0.0825 * jnp.sin(q_2) * jnp.cos(q_3) - 0.0825 * jnp.sin(q_4) * jnp.cos(q_2) + 0.384 * jnp.cos(q_2) * jnp.cos(q_4) + 0.316 * jnp.cos(q_2) + 0.33
    
    r_32 = -cq7 * (cq5 * sq2 * sq3 - sq5 * (cq2 * sq4 - cq3 * cq4 * sq2)) - sq7 * (cq6 * (cq5 * (cq2 * sq4 - cq3 * cq4 * sq2) + sq2 * sq3 * sq5) + sq6 * (cq2 * cq4 + cq3 * sq2 * sq4))
    r_33 = -cq6 * (cq2 * cq4 + cq3 * sq2 * sq4) + sq6 * (cq5 * (cq2 * sq4 - cq3 * cq4 * sq2) + sq2 * sq3 * sq5)
    r_31 = cq7 * (cq6 * (cq5 * (cq2 * sq4 - cq3 * cq4 * sq2) + sq2 * sq3 * sq5) + sq6 * (cq2 * cq4 + cq3 * sq2 * sq4)) - sq7 * (cq5 * sq2 * sq3 - sq5 * (cq2 * sq4 - cq3 * cq4 * sq2))

    # Position cost
    f_pos_cost = (x[-1] - x_fin) ** 2 + (y[-1] - y_fin) ** 2 + (z[-1] - z_fin) ** 2 + (x[0] - x_init) ** 2 + (y[0] - y_init) ** 2 + (z[0] - z_init) ** 2
    # Orientation cost
    f_orient_cost = jnp.sum((jnp.arctan2(r_32, r_33) - roll_des) ** 2 + (-jnp.arcsin(r_31) - pitch_des) ** 2)
    
    # Smoothness cost
    f_smoothness_vel = jnp.sum(jnp.dot(A_vel, q_1) ** 2) + jnp.sum(jnp.dot(A_vel, q_2) ** 2) + jnp.sum(jnp.dot(A_vel, q_3) ** 2) + jnp.sum(jnp.dot(A_vel, q_4) ** 2) + jnp.sum(jnp.dot(A_vel, q_5) ** 2) + jnp.sum(jnp.dot(A_vel, q_6) ** 2) + jnp.sum(jnp.dot(A_vel, q_7) ** 2)
    f_smoothness_acc = jnp.sum(jnp.dot(A_acc, q_1) ** 2) + jnp.sum(jnp.dot(A_acc, q_2) ** 2) + jnp.sum(jnp.dot(A_acc, q_3) ** 2) + jnp.sum(jnp.dot(A_acc, q_4) ** 2) + jnp.sum(jnp.dot(A_acc, q_5) ** 2) + jnp.sum(jnp.dot(A_acc, q_6) ** 2) + jnp.sum(jnp.dot(A_acc, q_7) ** 2)
    f_smoothness_jerk = jnp.sum(jnp.dot(A_jerk, q_1) ** 2) + jnp.sum(jnp.dot(A_jerk, q_2) ** 2) + jnp.sum(jnp.dot(A_jerk, q_3) ** 2) + jnp.sum(jnp.dot(A_jerk, q_4) ** 2) + jnp.sum(jnp.dot(A_jerk, q_5) ** 2) + jnp.sum(jnp.dot(A_jerk, q_6) ** 2) + jnp.sum(jnp.dot(A_jerk, q_7) ** 2)
    f_smoothness_cost = rho_vel * f_smoothness_vel + rho_acc * f_smoothness_acc + rho_jerk * f_smoothness_jerk

    cost = weight_orient * f_orient_cost + f_smoothness_cost + weight_pos * f_pos_cost
    return cost

def hessian_inp(fun):
    return jit(jacfwd(jacrev(fun)))

def hessian_params(fun):
    return jit(jacfwd(jacrev(fun), argnums=1))

F_YY_func = hessian_inp(cost_fun_jax)
F_XY_func = hessian_params(cost_fun_jax)
cost_function_jit = jit(cost_fun_jax)

def vmap_cost_function(a_batched, b_batched):
    return vmap(cost_function_jit)(a_batched, b_batched)

#### Computing perturbed solution
def compute_perturbed_solution(perturbed_params, res_new, nsteps=1):
    x_pred_list, cost_list = [], []
    x_diff = (perturbed_params - init_params).reshape(3, 1)
    x_pred = deepcopy(res.x)
    best_pred = deepcopy(x_pred)
    best_cost = cost_fun_jax(res.x, perturbed_params)
    
    eta_array = np.arange(0.05, 1.05, 0.05).reshape(-1, 1)
    ones_array = jnp.ones((20, 1))
    perturbed_params_batched = ones_array * perturbed_params

    cur_params = deepcopy(init_params)
    cur_diff = deepcopy(x_diff)
    x_pred_list.append(x_pred)
    cost_list.append(best_cost)
    
    for n in tqdm(range(nsteps)):
        F_YY = F_YY_func(x_pred, cur_params)
        F_XY = F_XY_func(x_pred, cur_params)
        F_YY_inv = jnp.linalg.inv(F_YY)
        
        dgx = jnp.matmul(-F_YY_inv, F_XY)
        dgx_prod = jnp.matmul(dgx, cur_diff).reshape(7 * num)
        dgx_prod_batched = eta_array * dgx_prod
        
        x_pred = x_pred + dgx_prod_batched
        x_pred = jnp.clip(x_pred, q_min_traj, q_max_traj)
        
        cost = vmap_cost_function(x_pred, perturbed_params_batched)
        min_idx = jnp.argmin(cost)
        min_cost = cost[min_idx]
        x_pred = x_pred[min_idx, :]
        x_pred_list.append(x_pred)        
        
        _, _, _, _, _, _, _, x_new, y_new, z_new, _, _, _, _ = get_trajectory(x_pred)
        cur_params = jnp.array([x_new[-1], y_new[-1], z_new[-1]])
        # print(cur_params.shape)
        cur_diff = (perturbed_params - cur_params).reshape(3, 1)
        # print(perturbed_params.shape, cur_params.shape)

        print("Iteration {:3d}, cost: {:8.4f}, min_idx: {}".format(n, min_cost, min_idx))
        if min_cost < best_cost:
            best_cost = min_cost
            best_pred = deepcopy(x_pred)
            cost_list.append(best_cost)
        else:
            break
    return x_pred, best_pred, best_cost, x_pred_list, cost_list

#### Generate argmin solutions and trajectories
num_samples = 6
flag = 1
cur_traj_cnt = 1

for num_iter in range(1, num_samples + 1):
    for sample_num in range(len(data['q_fin'][num_iter])):
        print("ITERATION {}, Cur traj num: {}".format(num_iter, cur_traj_cnt))
        print("Saving plots")
        save_trajectory_plots(cur_traj_cnt, x_init, y_init, z_init, x_fin, y_fin, z_fin, x, y, z, message='initial')
        save_q_vals_plots(cur_traj_cnt, q_1, q_2, q_3, q_4, q_5, q_6, q_7, message='initial')

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

        res_new.x = soln[cur_traj_cnt]
        initial_cost2 = cost_fun(res.x, perturbed_params)
        q_1_true, q_2_true, q_3_true, q_4_true, q_5_true, q_6_true, q_7_true, x_true, y_true, z_true, tr_true, ax1_true, ax2_true, ax3_true = get_trajectory(res_new.x)
    
        print("Saving Plots")
        save_trajectory_plots(cur_traj_cnt, x_init, y_init, z_init, x_fin_new, y_fin_new, z_fin_new, x_true, y_true, z_true, message='true_perturbed')
        save_q_vals_plots(cur_traj_cnt, q_1_true, q_2_true, q_3_true, q_4_true, q_5_true, q_6_true, q_7_true, message='true_perturbed')
        print("-"*50)

        norm_x = np.linalg.norm(x_fin_new - x_fin)
        norm_y = np.linalg.norm(y_fin_new - y_fin)
        norm_z = np.linalg.norm(z_fin_new - z_fin)
        total_dis = np.sqrt(norm_x ** 2 + norm_y ** 2 + norm_z ** 2)    

        x_sol = jnp.array(res.x)
        x_fin_new = jnp.array(x_fin_new)
        y_fin_new = jnp.array(y_fin_new)
        z_fin_new = jnp.array(z_fin_new)    

        if flag == 1:
            flag = 0
            
            a = np.float32(np.random.randn(*(res.x.shape)))
            b = np.float32(np.random.randn(*init_params.shape))
            ones_array = jnp.ones((20, 1))
            
            a_batched = ones_array * a
            b_batched = ones_array * b

            start_time = time.time()
            F_YY = F_YY_func(a, b)
            end_time = time.time()
            print("F_YY.shape: {}, First compilation time: {}".format(F_YY.shape, end_time - start_time))

            start_time = time.time()
            F_XY = F_XY_func(a, b)
            end_time = time.time()
            print("F_XY.shape: {}, First compilation time: {}".format(F_XY.shape, end_time - start_time))
            
            start_time = time.time()
            _out = cost_function_jit(a, b)
            end_time = time.time()
            print("_out: {}, First compilation time: {}".format(_out, end_time - start_time))

            start_time = time.time()
            _out = vmap_cost_function(a_batched, b_batched)
            end_time = time.time()
            print("_out.shape: {}, First compilation time: {}".format(_out.shape, end_time - start_time))
            
            start_time = time.time()
            F_YY_inv = jnp.linalg.inv(F_YY)
            end_time = time.time()
            print("F_YY_inv.shape: {}, First compilation time: {}".format(F_YY_inv.shape, end_time - start_time))
    
        print("Initial cost at old position: {}".format(initial_cost1))
        print("Initial cost at new position: {}".format(initial_cost2))

        start_time = time.time()
        start_process_time = time.process_time()
        x_pred, best_pred, best_cost, x_pred_list, cost_list = compute_perturbed_solution(perturbed_params, res_new, nsteps=50)
        end_process_time = time.process_time()
        end_time = time.time()
    
        actual_cost = cost_fun_jax(res_new.x, perturbed_params)
        print("Best cost: {}".format(best_cost))
        print("Actual Cost: {}".format(actual_cost))
        print("Argmin Time Taken: {}".format(end_time - start_time))
        print("Argmin process time: {}".format(end_process_time - start_process_time))
        print("Norm bw actual and predicted solution: {:7.4f}".format(np.linalg.norm(best_pred - res_new.x)))
        print("Min diff: {:7.4f}".format(np.min(np.abs(best_pred - res_new.x))))
        print("Max diff: {:7.4f}".format(np.max(np.abs(best_pred - res_new.x))))

        q_1_pred, q_2_pred, q_3_pred, q_4_pred, q_5_pred, q_6_pred, q_7_pred, x_pred, y_pred, z_pred, tr_pred, ax1_pred, ax2_pred, \
            ax3_pred = get_trajectory(best_pred)

        norm_x_traj = np.linalg.norm(x_pred - x_true)
        norm_y_traj = np.linalg.norm(y_pred - y_true)
        norm_z_traj = np.linalg.norm(z_pred - z_true)
        f_pos_cost_pred, f_orient_cost_pred, f_smoothness_vel_pred, f_smoothness_acc_pred, f_smoothness_jerk_pred, roll_pred, pitch_pred = \
            get_cost_components(best_pred, perturbed_params)

        actual_cost = cost_fun_jax(res_new.x, perturbed_params)
        f_pos_cost_actual, f_orient_cost_actual, f_smoothness_vel_actual, f_smoothness_acc_actual, f_smoothness_jerk_actual, roll_actual, \
            pitch_actual = get_cost_components(res_new.x, perturbed_params)

        roll_cost = jnp.max(abs(roll_actual - roll_pred))
        pitch_cost = jnp.max(abs(pitch_actual - pitch_pred))

        cost_dict = {
            # Initial positions
            "x_init": str(x_init),
            "y_init": str(y_init),
            "z_init": str(z_init),
            # Final positions
            "x_fin": str(x_fin),
            "y_fin": str(y_fin),
            "z_fin": str(z_fin),
            # Perturbed final positions
            "x_fin_new": str(x_fin_new),
            "y_fin_new": str(y_fin_new),
            "z_fin_new": str(z_fin_new),
            # Perturbation norms
            "total_perturbation": str(total_dis),
            "norm_x_perturbation": str(norm_x),
            "norm_y_perturbation": str(norm_y),
            "norm_z_perturbation": str(norm_z),
            # Solver costs at new and old positions
            "initial_cost_old_pos": str(initial_cost1),
            "initial_cost_new_pos": str(initial_cost2),
            # Norm of end effector x, y, z positions
            'norm_x_traj': str(norm_x_traj),
            'norm_y_traj': str(norm_y_traj),
            'norm_z_traj': str(norm_z_traj),
            # Norm of solver and argmin solution
            'norm_sol': str(np.linalg.norm(best_pred - res_new.x)),
            # Argmin solution cost
            'best_cost': str(best_cost.item()),
            # Actual solver cost
            'actual_cost': str(actual_cost.item()),
            # Max difference bw roll & pitch of solver and argmin solution
            'roll_cost': str(roll_cost.item()),
            'pitch_cost': str(pitch_cost.item()),
            # Prediction costs
            'f_pos_cost_pred': str(f_pos_cost_pred.item()),
            'f_orient_cost_pred': str(f_orient_cost_pred.item()),
            'f_smoothness_vel_pred': str(f_smoothness_vel_pred.item()),
            'f_smoothness_acc_pred': str(f_smoothness_acc_pred.item()),
            'f_smoothness_jerk_pred': str(f_smoothness_jerk_pred.item()),
            # Actual costs
            'f_pos_cost_actual': str(f_pos_cost_actual.item()),
            'f_orient_cost_actual': str(f_orient_cost_actual.item()),
            'f_smoothness_vel_actual': str(f_smoothness_vel_actual.item()),
            'f_smoothness_acc_actual': str(f_smoothness_acc_actual.item()),
            'f_smoothness_jerk_actual': str(f_smoothness_jerk_actual.item()),
            # Timing data
            'argmin_time': str(end_time - start_time),
            'argmin_process_time': str(end_process_time - start_process_time),
            'solver_time': str(end_solver_time - start_solver_time),
            'solver_process_time': str(end_solver_process_time - start_solver_process_time)
        }

        print("Saving plots")
        save_trajectory_plots_comparison(cur_traj_cnt, x_init, y_init, z_init, x_fin, y_fin, z_fin, x, y, z, x_fin_new, y_fin_new, \
            z_fin_new, x_true, y_true, z_true, x_pred, y_pred, z_pred, total_dis, cost_dict, str(best_cost.item()), \
                message='comparison')
        
        save_q_vals_plots_compare(cur_traj_cnt, q_1_true, q_2_true, q_3_true, q_4_true, q_5_true, q_6_true, q_7_true, q_1_pred, \
            q_2_pred, q_3_pred, q_4_pred, q_5_pred, q_6_pred, q_7_pred, total_dis, cost_dict, message='comparison')

        save_roll_plots(cur_traj_cnt, roll_pred, roll_actual, message='comparison')
        save_pitch_plots(cur_traj_cnt, pitch_pred, pitch_actual, message='comparison')

        json_path = os.path.join(args.results_path, 'traj_{}'.format(cur_pos + 1), 'sample_{}'.format(cur_traj_cnt), 'soln.json')
        with open(json_path, 'w') as f:
            json.dump(cost_dict, f, indent=4)
        
        numpy_dict = {
            "q_init": q_init,
            "q_fin": q_fin,
            "pos_init": [x_init, y_init, z_init],
            "pos_fin": [x_fin, y_fin, z_fin],
            "pos_fin_new": [x_fin_new, y_fin_new, z_fin_new],
            "pos_init_new": None,
            "q": res.x.reshape(7, 50).T,
            "q_solver": res_new.x.reshape(7, 50).T,
            "q_argmin": best_pred.reshape(7, 50).T,
            "x_pred_list": x_pred_list,
            "cost_list": cost_list
        }
        np_path = os.path.join(args.results_path, 'traj_{}'.format(cur_pos + 1), 'sample_{}'.format(cur_traj_cnt), 'traj_data.npy')
        np.save(np_path, numpy_dict)
        print("-"*100)
        
        cur_traj_cnt += 1