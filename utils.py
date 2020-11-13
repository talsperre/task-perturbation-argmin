import time

import jax.numpy as jnp
from autograd import grad
import autograd.numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from jax import jit, jacfwd, jacrev
from scipy.optimize import minimize, Bounds


def stack_y(q1, q2, q3, q4, q5, q6, q7):
    return np.concatenate((q1, q2, q3, q4, q5, q6, q7))


def unstack_y(y):
    n = y.shape[0]//7
    q1 = y[0:n]
    q2 = y[1*n:2*n]
    q3 = y[2*n:3*n]
    q4 = y[3*n:4*n]
    q5 = y[4*n:5*n]
    q6 = y[5*n:6*n]
    q7 = y[6*n:7*n]
    return q1, q2, q3, q4, q5, q6, q7


def fk_franka_traj(q):

    q1, q2, q3, q4, q5, q6, q7 = unstack_y(q)

    x = -0.107*(((np.sin(q1)*np.sin(q3) - np.cos(q1)*np.cos(q2)*np.cos(q3))*np.cos(q4) - np.sin(q2)*np.sin(q4)*np.cos(q1))*np.cos(q5) + (np.sin(q1)*np.cos(q3) + np.sin(q3)*np.cos(q1)*np.cos(q2))*np.sin(q5))*np.sin(q6) - 0.088*(((np.sin(q1)*np.sin(q3) - np.cos(q1)*np.cos(q2)*np.cos(q3))*np.cos(q4) - np.sin(q2)*np.sin(q4)*np.cos(q1))*np.cos(q5) + (np.sin(q1)*np.cos(q3) + np.sin(q3)*np.cos(q1)*np.cos(q2))*np.sin(q5))*np.cos(q6) + 0.088*((np.sin(q1)*np.sin(q3) - np.cos(q1)*np.cos(q2)*np.cos(q3))*np.sin(q4) + np.sin(q2)*np.cos(q1)*np.cos(q4))*np.sin(q6) - 0.107*((np.sin(q1)*np.sin(q3) - np.cos(q1)*np.cos(q2)*np.cos(q3))*np.sin(q4) + np.sin(q2)*np.cos(q1)*np.cos(q4))*np.cos(q6) + 0.384*(np.sin(q1)*np.sin(q3) - np.cos(q1)*np.cos(q2)*np.cos(q3))*np.sin(q4) + 0.0825*(np.sin(q1)*np.sin(q3) - np.cos(q1)*np.cos(q2)*np.cos(q3))*np.cos(q4) - 0.0825*np.sin(q1)*np.sin(q3) - 0.0825*np.sin(q2)*np.sin(q4)*np.cos(q1) + 0.384*np.sin(q2)*np.cos(q1)*np.cos(q4) + 0.316*np.sin(q2)*np.cos(q1) + 0.0825*np.cos(q1)*np.cos(q2)*np.cos(q3)

    y = 0.107*(((np.sin(q1)*np.cos(q2)*np.cos(q3) + np.sin(q3)*np.cos(q1))*np.cos(q4) + np.sin(q1)*np.sin(q2)*np.sin(q4))*np.cos(q5) - (np.sin(q1)*np.sin(q3)*np.cos(q2) - np.cos(q1)*np.cos(q3))*np.sin(q5))*np.sin(q6) + 0.088*(((np.sin(q1)*np.cos(q2)*np.cos(q3) + np.sin(q3)*np.cos(q1))*np.cos(q4) + np.sin(q1)*np.sin(q2)*np.sin(q4))*np.cos(q5) - (np.sin(q1)*np.sin(q3)*np.cos(q2) - np.cos(q1)*np.cos(q3))*np.sin(q5))*np.cos(q6) - 0.088*((np.sin(q1)*np.cos(q2)*np.cos(q3) + np.sin(q3)*np.cos(q1))*np.sin(q4) - np.sin(q1)*np.sin(q2)*np.cos(q4))*np.sin(q6) + 0.107*((np.sin(q1)*np.cos(q2)*np.cos(q3) + np.sin(q3)*np.cos(q1))*np.sin(q4) - np.sin(q1)*np.sin(q2)*np.cos(q4))*np.cos(q6) - 0.384*(np.sin(q1)*np.cos(q2)*np.cos(q3) + np.sin(q3)*np.cos(q1))*np.sin(q4) - 0.0825*(np.sin(q1)*np.cos(q2)*np.cos(q3) + np.sin(q3)*np.cos(q1))*np.cos(q4) - 0.0825*np.sin(q1)*np.sin(q2)*np.sin(q4) + 0.384*np.sin(q1)*np.sin(q2)*np.cos(q4) + 0.316*np.sin(q1)*np.sin(q2) + 0.0825*np.sin(q1)*np.cos(q2)*np.cos(q3) + 0.0825*np.sin(q3)*np.cos(q1)

    z = -0.107*((np.sin(q2)*np.cos(q3)*np.cos(q4) - np.sin(q4)*np.cos(q2))*np.cos(q5) - np.sin(q2)*np.sin(q3)*np.sin(q5))*np.sin(q6) - 0.088*((np.sin(q2)*np.cos(q3)*np.cos(q4) - np.sin(q4)*np.cos(q2))*np.cos(q5) - np.sin(q2)*np.sin(q3)*np.sin(q5))*np.cos(q6) + 0.088*(np.sin(q2)*np.sin(q4)*np.cos(q3) + np.cos(q2)*np.cos(q4))*np.sin(q6) - 0.107*(np.sin(q2)*np.sin(q4)*np.cos(q3) + np.cos(q2)*np.cos(q4))*np.cos(q6) + 0.384*np.sin(q2)*np.sin(q4)*np.cos(q3) + 0.0825*np.sin(q2)*np.cos(q3)*np.cos(q4) - 0.0825*np.sin(q2)*np.cos(q3) - 0.0825*np.sin(q4)*np.cos(q2) + 0.384*np.cos(q2)*np.cos(q4) + 0.316*np.cos(q2) + 0.33

    tr = np.sin(q2)*np.sin(q3)*np.sin(q5)*np.sin(q6) + np.sin(q2)*np.sin(q4)*np.sin(q5)*np.sin(q1 + q7) - np.sin(q2)*np.sin(q4)*np.cos(q3)*np.cos(q6) + np.sin(q2)*np.sin(q4)*np.cos(q5)*np.cos(q6)*np.cos(q1 + q7) - np.sin(q2)*np.sin(q6)*np.cos(q3)*np.cos(q4)*np.cos(q5) + np.sin(q2)*np.sin(q6)*np.cos(q4)*np.cos(q1 + q7) + np.sin(q3)*np.sin(q4)*np.sin(q6)*np.sin(q1 + q7) - np.sin(q3)*np.sin(q5)*np.cos(q2)*np.cos(q6)*np.cos(q1 + q7) + np.sin(q3)*np.sin(q5)*np.cos(q4)*np.cos(q1 + q7) + np.sin(q3)*np.sin(q1 + q7)*np.cos(q2)*np.cos(q5) - np.sin(q3)*np.sin(q1 + q7)*np.cos(q4)*np.cos(q5)*np.cos(q6) - np.sin(q4)*np.sin(q6)*np.cos(q2)*np.cos(q3)*np.cos(q1 + q7) + np.sin(q4)*np.sin(q6)*np.cos(q2)*np.cos(q5) + np.sin(q5)*np.sin(q1 + q7)*np.cos(q2)*np.cos(q3)*np.cos(q4) - np.sin(q5)*np.sin(q1 + q7)*np.cos(q3)*np.cos(q6) + np.cos(q2)*np.cos(q3)*np.cos(q4)*np.cos(q5)*np.cos(q6)*np.cos(q1 + q7) - np.cos(q2)*np.cos(q4)*np.cos(q6) - np.cos(q3)*np.cos(q5)*np.cos(q1 + q7)

    ax1 = -0.5*(((np.sin(q1)*np.cos(q2)*np.cos(q3) + np.sin(q3)*np.cos(q1))*np.cos(q4) + np.sin(q1)*np.sin(q2)*np.sin(q4))*np.cos(q5) - (np.sin(q1)*np.sin(q3)*np.cos(q2) - np.cos(q1)*np.cos(q3))*np.sin(q5))*np.sin(q6) + 0.5*(((np.sin(q2)*np.cos(q3)*np.cos(q4) - np.sin(q4)*np.cos(q2))*np.cos(q5) - np.sin(q2)*np.sin(q3)*np.sin(q5))*np.cos(q6) - (np.sin(q2)*np.sin(q4)*np.cos(q3) + np.cos(q2)*np.cos(q4))*np.sin(q6))*np.sin(q7) - 0.5*((np.sin(q1)*np.cos(q2)*np.cos(q3) + np.sin(q3)*np.cos(q1))*np.sin(q4) - np.sin(q1)*np.sin(q2)*np.cos(q4))*np.cos(q6) - 0.5*((np.sin(q2)*np.cos(q3)*np.cos(q4) - np.sin(q4)*np.cos(q2))*np.sin(q5) + np.sin(q2)*np.sin(q3)*np.cos(q5))*np.cos(q7)

    ax2 = -0.5*(((np.sin(q1)*np.sin(q3) - np.cos(q1)*np.cos(q2)*np.cos(q3))*np.cos(q4) - np.sin(q2)*np.sin(q4)*np.cos(q1))*np.cos(q5) + (np.sin(q1)*np.cos(q3) + np.sin(q3)*np.cos(q1)*np.cos(q2))*np.sin(q5))*np.sin(q6) + 0.5*(((np.sin(q2)*np.cos(q3)*np.cos(q4) - np.sin(q4)*np.cos(q2))*np.cos(q5) - np.sin(q2)*np.sin(q3)*np.sin(q5))*np.cos(q6) - (np.sin(q2)*np.sin(q4)*np.cos(q3) + np.cos(q2)*np.cos(q4))*np.sin(q6))*np.cos(q7) - 0.5*((np.sin(q1)*np.sin(q3) - np.cos(q1)*np.cos(q2)*np.cos(q3))*np.sin(q4) + np.sin(q2)*np.cos(q1)*np.cos(q4))*np.cos(q6) + 0.5*((np.sin(q2)*np.cos(q3)*np.cos(q4) - np.sin(q4)*np.cos(q2))*np.sin(q5) + np.sin(q2)*np.sin(q3)*np.cos(q5))*np.sin(q7)

    ax3 = -0.5*np.sin(q2)*np.sin(q4)*np.sin(q5)*np.cos(q1 + q7) + 0.5*np.sin(q2)*np.sin(q4)*np.sin(q1 + q7)*np.cos(q5)*np.cos(q6) + 0.5*np.sin(q2)*np.sin(q6)*np.sin(q1 + q7)*np.cos(q4) - 0.5*np.sin(q3)*np.sin(q4)*np.sin(q6)*np.cos(q1 + q7) - 0.5*np.sin(q3)*np.sin(q5)*np.sin(q1 + q7)*np.cos(q2)*np.cos(q6) + 0.5*np.sin(q3)*np.sin(q5)*np.sin(q1 + q7)*np.cos(q4) - 0.5*np.sin(q3)*np.cos(q2)*np.cos(q5)*np.cos(q1 + q7) + 0.5*np.sin(q3)*np.cos(q4)*np.cos(q5)*np.cos(q6)*np.cos(q1 + q7) - 0.5*np.sin(q4)*np.sin(q6)*np.sin(q1 + q7)*np.cos(q2)*np.cos(q3) - 0.5*np.sin(q5)*np.cos(q2)*np.cos(q3)*np.cos(q4)*np.cos(q1 + q7) + 0.5*np.sin(q5)*np.cos(q3)*np.cos(q6)*np.cos(q1 + q7) + 0.5*np.sin(q1 + q7)*np.cos(q2)*np.cos(q3)*np.cos(q4)*np.cos(q5)*np.cos(q6) - 0.5*np.sin(q1 + q7)*np.cos(q3)*np.cos(q5)

    return x, y, z, tr, ax1, ax2, ax3


def fk_franka_traj_jax(q):

    q1, q2, q3, q4, q5, q6, q7 = unstack_y(q)

    x = -0.107*(((jnp.sin(q1)*jnp.sin(q3) - jnp.cos(q1)*jnp.cos(q2)*jnp.cos(q3))*jnp.cos(q4) - jnp.sin(q2)*jnp.sin(q4)*jnp.cos(q1))*jnp.cos(q5) + (jnp.sin(q1)*jnp.cos(q3) + jnp.sin(q3)*jnp.cos(q1)*jnp.cos(q2))*jnp.sin(q5))*jnp.sin(q6) - 0.088*(((jnp.sin(q1)*jnp.sin(q3) - jnp.cos(q1)*jnp.cos(q2)*jnp.cos(q3))*jnp.cos(q4) - jnp.sin(q2)*jnp.sin(q4)*jnp.cos(q1))*jnp.cos(q5) + (jnp.sin(q1)*jnp.cos(q3) + jnp.sin(q3)*jnp.cos(q1)*jnp.cos(q2))*jnp.sin(q5))*jnp.cos(q6) + 0.088*((jnp.sin(q1)*jnp.sin(q3) - jnp.cos(q1)*jnp.cos(q2)*jnp.cos(q3))*jnp.sin(q4) + jnp.sin(q2)*jnp.cos(q1)*jnp.cos(q4))*jnp.sin(q6) - 0.107*((jnp.sin(q1)*jnp.sin(q3) - jnp.cos(q1)*jnp.cos(q2)*jnp.cos(q3))*jnp.sin(q4) + jnp.sin(q2)*jnp.cos(q1)*jnp.cos(q4))*jnp.cos(q6) + 0.384*(jnp.sin(q1)*jnp.sin(q3) - jnp.cos(q1)*jnp.cos(q2)*jnp.cos(q3))*jnp.sin(q4) + 0.0825*(jnp.sin(q1)*jnp.sin(q3) - jnp.cos(q1)*jnp.cos(q2)*jnp.cos(q3))*jnp.cos(q4) - 0.0825*jnp.sin(q1)*jnp.sin(q3) - 0.0825*jnp.sin(q2)*jnp.sin(q4)*jnp.cos(q1) + 0.384*jnp.sin(q2)*jnp.cos(q1)*jnp.cos(q4) + 0.316*jnp.sin(q2)*jnp.cos(q1) + 0.0825*jnp.cos(q1)*jnp.cos(q2)*jnp.cos(q3)

    y = 0.107*(((jnp.sin(q1)*jnp.cos(q2)*jnp.cos(q3) + jnp.sin(q3)*jnp.cos(q1))*jnp.cos(q4) + jnp.sin(q1)*jnp.sin(q2)*jnp.sin(q4))*jnp.cos(q5) - (jnp.sin(q1)*jnp.sin(q3)*jnp.cos(q2) - jnp.cos(q1)*jnp.cos(q3))*jnp.sin(q5))*jnp.sin(q6) + 0.088*(((jnp.sin(q1)*jnp.cos(q2)*jnp.cos(q3) + jnp.sin(q3)*jnp.cos(q1))*jnp.cos(q4) + jnp.sin(q1)*jnp.sin(q2)*jnp.sin(q4))*jnp.cos(q5) - (jnp.sin(q1)*jnp.sin(q3)*jnp.cos(q2) - jnp.cos(q1)*jnp.cos(q3))*jnp.sin(q5))*jnp.cos(q6) - 0.088*((jnp.sin(q1)*jnp.cos(q2)*jnp.cos(q3) + jnp.sin(q3)*jnp.cos(q1))*jnp.sin(q4) - jnp.sin(q1)*jnp.sin(q2)*jnp.cos(q4))*jnp.sin(q6) + 0.107*((jnp.sin(q1)*jnp.cos(q2)*jnp.cos(q3) + jnp.sin(q3)*jnp.cos(q1))*jnp.sin(q4) - jnp.sin(q1)*jnp.sin(q2)*jnp.cos(q4))*jnp.cos(q6) - 0.384*(jnp.sin(q1)*jnp.cos(q2)*jnp.cos(q3) + jnp.sin(q3)*jnp.cos(q1))*jnp.sin(q4) - 0.0825*(jnp.sin(q1)*jnp.cos(q2)*jnp.cos(q3) + jnp.sin(q3)*jnp.cos(q1))*jnp.cos(q4) - 0.0825*jnp.sin(q1)*jnp.sin(q2)*jnp.sin(q4) + 0.384*jnp.sin(q1)*jnp.sin(q2)*jnp.cos(q4) + 0.316*jnp.sin(q1)*jnp.sin(q2) + 0.0825*jnp.sin(q1)*jnp.cos(q2)*jnp.cos(q3) + 0.0825*jnp.sin(q3)*jnp.cos(q1)

    z = -0.107*((jnp.sin(q2)*jnp.cos(q3)*jnp.cos(q4) - jnp.sin(q4)*jnp.cos(q2))*jnp.cos(q5) - jnp.sin(q2)*jnp.sin(q3)*jnp.sin(q5))*jnp.sin(q6) - 0.088*((jnp.sin(q2)*jnp.cos(q3)*jnp.cos(q4) - jnp.sin(q4)*jnp.cos(q2))*jnp.cos(q5) - jnp.sin(q2)*jnp.sin(q3)*jnp.sin(q5))*jnp.cos(q6) + 0.088*(jnp.sin(q2)*jnp.sin(q4)*jnp.cos(q3) + jnp.cos(q2)*jnp.cos(q4))*jnp.sin(q6) - 0.107*(jnp.sin(q2)*jnp.sin(q4)*jnp.cos(q3) + jnp.cos(q2)*jnp.cos(q4))*jnp.cos(q6) + 0.384*jnp.sin(q2)*jnp.sin(q4)*jnp.cos(q3) + 0.0825*jnp.sin(q2)*jnp.cos(q3)*jnp.cos(q4) - 0.0825*jnp.sin(q2)*jnp.cos(q3) - 0.0825*jnp.sin(q4)*jnp.cos(q2) + 0.384*jnp.cos(q2)*jnp.cos(q4) + 0.316*jnp.cos(q2) + 0.33

    tr = jnp.sin(q2)*jnp.sin(q3)*jnp.sin(q5)*jnp.sin(q6) + jnp.sin(q2)*jnp.sin(q4)*jnp.sin(q5)*jnp.sin(q1 + q7) - jnp.sin(q2)*jnp.sin(q4)*jnp.cos(q3)*jnp.cos(q6) + jnp.sin(q2)*jnp.sin(q4)*jnp.cos(q5)*jnp.cos(q6)*jnp.cos(q1 + q7) - jnp.sin(q2)*jnp.sin(q6)*jnp.cos(q3)*jnp.cos(q4)*jnp.cos(q5) + jnp.sin(q2)*jnp.sin(q6)*jnp.cos(q4)*jnp.cos(q1 + q7) + jnp.sin(q3)*jnp.sin(q4)*jnp.sin(q6)*jnp.sin(q1 + q7) - jnp.sin(q3)*jnp.sin(q5)*jnp.cos(q2)*jnp.cos(q6)*jnp.cos(q1 + q7) + jnp.sin(q3)*jnp.sin(q5)*jnp.cos(q4)*jnp.cos(q1 + q7) + jnp.sin(q3)*jnp.sin(q1 + q7)*jnp.cos(q2)*jnp.cos(q5) - jnp.sin(q3)*jnp.sin(q1 + q7)*jnp.cos(q4)*jnp.cos(q5)*jnp.cos(q6) - jnp.sin(q4)*jnp.sin(q6)*jnp.cos(q2)*jnp.cos(q3)*jnp.cos(q1 + q7) + jnp.sin(q4)*jnp.sin(q6)*jnp.cos(q2)*jnp.cos(q5) + jnp.sin(q5)*jnp.sin(q1 + q7)*jnp.cos(q2)*jnp.cos(q3)*jnp.cos(q4) - jnp.sin(q5)*jnp.sin(q1 + q7)*jnp.cos(q3)*jnp.cos(q6) + jnp.cos(q2)*jnp.cos(q3)*jnp.cos(q4)*jnp.cos(q5)*jnp.cos(q6)*jnp.cos(q1 + q7) - jnp.cos(q2)*jnp.cos(q4)*jnp.cos(q6) - jnp.cos(q3)*jnp.cos(q5)*jnp.cos(q1 + q7)

    ax1 = -0.5*(((jnp.sin(q1)*jnp.cos(q2)*jnp.cos(q3) + jnp.sin(q3)*jnp.cos(q1))*jnp.cos(q4) + jnp.sin(q1)*jnp.sin(q2)*jnp.sin(q4))*jnp.cos(q5) - (jnp.sin(q1)*jnp.sin(q3)*jnp.cos(q2) - jnp.cos(q1)*jnp.cos(q3))*jnp.sin(q5))*jnp.sin(q6) + 0.5*(((jnp.sin(q2)*jnp.cos(q3)*jnp.cos(q4) - jnp.sin(q4)*jnp.cos(q2))*jnp.cos(q5) - jnp.sin(q2)*jnp.sin(q3)*jnp.sin(q5))*jnp.cos(q6) - (jnp.sin(q2)*jnp.sin(q4)*jnp.cos(q3) + jnp.cos(q2)*jnp.cos(q4))*jnp.sin(q6))*jnp.sin(q7) - 0.5*((jnp.sin(q1)*jnp.cos(q2)*jnp.cos(q3) + jnp.sin(q3)*jnp.cos(q1))*jnp.sin(q4) - jnp.sin(q1)*jnp.sin(q2)*jnp.cos(q4))*jnp.cos(q6) - 0.5*((jnp.sin(q2)*jnp.cos(q3)*jnp.cos(q4) - jnp.sin(q4)*jnp.cos(q2))*jnp.sin(q5) + jnp.sin(q2)*jnp.sin(q3)*jnp.cos(q5))*jnp.cos(q7)

    ax2 = -0.5*(((jnp.sin(q1)*jnp.sin(q3) - jnp.cos(q1)*jnp.cos(q2)*jnp.cos(q3))*jnp.cos(q4) - jnp.sin(q2)*jnp.sin(q4)*jnp.cos(q1))*jnp.cos(q5) + (jnp.sin(q1)*jnp.cos(q3) + jnp.sin(q3)*jnp.cos(q1)*jnp.cos(q2))*jnp.sin(q5))*jnp.sin(q6) + 0.5*(((jnp.sin(q2)*jnp.cos(q3)*jnp.cos(q4) - jnp.sin(q4)*jnp.cos(q2))*jnp.cos(q5) - jnp.sin(q2)*jnp.sin(q3)*jnp.sin(q5))*jnp.cos(q6) - (jnp.sin(q2)*jnp.sin(q4)*jnp.cos(q3) + jnp.cos(q2)*jnp.cos(q4))*jnp.sin(q6))*jnp.cos(q7) - 0.5*((jnp.sin(q1)*jnp.sin(q3) - jnp.cos(q1)*jnp.cos(q2)*jnp.cos(q3))*jnp.sin(q4) + jnp.sin(q2)*jnp.cos(q1)*jnp.cos(q4))*jnp.cos(q6) + 0.5*((jnp.sin(q2)*jnp.cos(q3)*jnp.cos(q4) - jnp.sin(q4)*jnp.cos(q2))*jnp.sin(q5) + jnp.sin(q2)*jnp.sin(q3)*jnp.cos(q5))*jnp.sin(q7)

    ax3 = -0.5*jnp.sin(q2)*jnp.sin(q4)*jnp.sin(q5)*jnp.cos(q1 + q7) + 0.5*jnp.sin(q2)*jnp.sin(q4)*jnp.sin(q1 + q7)*jnp.cos(q5)*jnp.cos(q6) + 0.5*jnp.sin(q2)*jnp.sin(q6)*jnp.sin(q1 + q7)*jnp.cos(q4) - 0.5*jnp.sin(q3)*jnp.sin(q4)*jnp.sin(q6)*jnp.cos(q1 + q7) - 0.5*jnp.sin(q3)*jnp.sin(q5)*jnp.sin(q1 + q7)*jnp.cos(q2)*jnp.cos(q6) + 0.5*jnp.sin(q3)*jnp.sin(q5)*jnp.sin(q1 + q7)*jnp.cos(q4) - 0.5*jnp.sin(q3)*jnp.cos(q2)*jnp.cos(q5)*jnp.cos(q1 + q7) + 0.5*jnp.sin(q3)*jnp.cos(q4)*jnp.cos(q5)*jnp.cos(q6)*jnp.cos(q1 + q7) - 0.5*jnp.sin(q4)*jnp.sin(q6)*jnp.sin(q1 + q7)*jnp.cos(q2)*jnp.cos(q3) - 0.5*jnp.sin(q5)*jnp.cos(q2)*jnp.cos(q3)*jnp.cos(q4)*jnp.cos(q1 + q7) + 0.5*jnp.sin(q5)*jnp.cos(q3)*jnp.cos(q6)*jnp.cos(q1 + q7) + 0.5*jnp.sin(q1 + q7)*jnp.cos(q2)*jnp.cos(q3)*jnp.cos(q4)*jnp.cos(q5)*jnp.cos(q6) - 0.5*jnp.sin(q1 + q7)*jnp.cos(q3)*jnp.cos(q5)

    return x, y, z, tr, ax1, ax2, ax3


def plot_dummy(q):
    x, y, z, _, _, _, _ = fk_franka_traj(q)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, '-o', linewidth=2.0, markersize=3.0, alpha=0.3)
    ax.set_xlim3d(np.min(x)-0.3, np.max(x)+0.3)
    ax.set_ylim3d(np.min(y)-0.3, np.max(y)+0.3)
    ax.set_zlim3d(np.min(z)-0.3, np.max(z)+0.3)
    ax.set_title('End Effector trajectory of a manipulator')
    plt.show()


def plot_trajectory(q, q_new, q_pred, q_init, q_fin, q_init_new, q_fin_new, folder=None):
    x_init, y_init, z_init, _, _, _, _ = fk_franka_traj(q_init)
    x_fin, y_fin, z_fin, _, _, _, _ = fk_franka_traj(q_fin)
    x_init_new, y_init_new, z_init_new, _, _, _, _ = fk_franka_traj(q_init_new)
    x_fin_new, y_fin_new, z_fin_new, _, _, _, _ = fk_franka_traj(q_fin_new)

    x, y, z, _, _, _, _ = fk_franka_traj(q)
    x_new, y_new, z_new, _, _, _, _ = fk_franka_traj(q_new)
    x_pred, y_pred, z_pred, _, _, _, _ = fk_franka_traj(q_pred)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x, y, z, '-o', linewidth=2.0, markersize=3.0, alpha=0.3)
    ax.plot(x_init, y_init, z_init, 'om', markersize=10, alpha=0.4)
    ax.plot(x_fin, y_fin, z_fin, 'og', markersize=10, alpha=0.4)
    ax.plot(x_init_new, y_init_new, z_init_new, 'om', markersize=10)
    ax.plot(x_fin_new, y_fin_new, z_fin_new, 'og', markersize=10)

    ax.plot(x_new, y_new, z_new, '-ok', linewidth=3.0, markersize=6.0)
    ax.plot(x_pred, y_pred, z_pred, '-o', linewidth=3.0, markersize=6.0)
    ax.set_xlim3d(np.min(x)-0.3, np.max(x)+0.3)
    ax.set_ylim3d(np.min(y)-0.3, np.max(y)+0.3)
    ax.set_zlim3d(np.min(z)-0.3, np.max(z)+0.3)
    ax.set_title('End Effector trajectory of a manipulator')

    if folder is None:
        plt.show()
    else:
        plt.savefig(folder+'/traj.png')
        plt.close()


def plot_trajectory_traj(q, q_new, q_pred, folder=None):

    x, y, z, _, _, _, _ = fk_franka_traj(q)
    x_new, y_new, z_new, _, _, _, _ = fk_franka_traj(q_new)
    x_pred, y_pred, z_pred, _, _, _, _ = fk_franka_traj(q_pred)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x, y, z, '-o', linewidth=2.0, markersize=3.0, alpha=0.3)
    ax.plot(x_new, y_new, z_new, '-ok', linewidth=3.0, markersize=6.0)
    ax.plot(x_pred, y_pred, z_pred, '-o', linewidth=3.0, markersize=6.0)

    ax.set_xlim3d(np.min(x)-0.3, np.max(x)+0.3)
    ax.set_ylim3d(np.min(y)-0.3, np.max(y)+0.3)
    ax.set_zlim3d(np.min(z)-0.3, np.max(z)+0.3)
    ax.set_title('End Effector trajectory of a manipulator')

    if folder is None:
        plt.show()
    else:
        plt.savefig(folder+'/traj.png')
        plt.close()


def plot_trajectory_mid(q, q_new, q_pred, q_init, q_fin, x_mid, y_mid, z_mid, x_mid_new, y_mid_new, z_mid_new, folder=None):
    x_init, y_init, z_init, _, _, _, _ = fk_franka_traj(q_init)
    x_fin, y_fin, z_fin, _, _, _, _ = fk_franka_traj(q_fin)

    x, y, z, _, _, _, _ = fk_franka_traj(q)
    x_new, y_new, z_new, _, _, _, _ = fk_franka_traj(q_new)
    x_pred, y_pred, z_pred, _, _, _, _ = fk_franka_traj(q_pred)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x, y, z, '-o', linewidth=2.0, markersize=3.0, alpha=0.3)
    ax.plot(x_init, y_init, z_init, 'om', markersize=10, alpha=0.4)
    ax.plot(x_fin, y_fin, z_fin, 'ob', markersize=10, alpha=0.4)
    ax.plot([x_mid], [y_mid], [z_mid], 'og', markersize=10, alpha=0.4)
    ax.plot([x_mid_new], [y_mid_new], [z_mid_new], 'og', markersize=10)

    ax.plot(x_new, y_new, z_new, '-ok', linewidth=3.0, markersize=6.0)
    ax.plot(x_pred, y_pred, z_pred, '-o', linewidth=3.0, markersize=6.0)
    # ax.set_xlim3d(-1.0, 1.0)
    # ax.set_ylim3d(-1.0, 1.0)
    # ax.set_zlim3d(-0.3, 1.2)
    ax.set_xlim3d(np.min(x)-0.3, np.max(x)+0.3)
    ax.set_ylim3d(np.min(y)-0.3, np.max(y)+0.3)
    ax.set_zlim3d(np.min(z)-0.3, np.max(z)+0.3)
    ax.set_title('End Effector trajectory of a manipulator')

    if folder is None:
        plt.show()
    else:
        # plt.show()
        plt.savefig(folder+'/traj.png')
        plt.close()

    
def plot_end_effector_angles(q_new, q_pred, folder=None):
    roll_new, pitch_new = get_roll_pitch(q_new)
    roll_pred, pitch_pred = get_roll_pitch(q_pred)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    ax.plot(roll_new, '-r')
    ax.plot(pitch_new, '-b')
    ax.plot(roll_pred, '--r')
    ax.plot(pitch_pred, '--b')
    ax.grid()
    ax.set_title('roll and pitch of end effector')

    if folder is None:
        plt.show()
    else:
        plt.savefig(folder+'/end_effector_angles.png')
        plt.close()


def plot_eta_vs_cost(cost_vec, eta, name=None, folder=None):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    ax.grid()
    ax.plot(eta, cost_vec, '-ro')
    ax.set_title('Eta vs cost')
    if folder is None:
        plt.show()
    else:
        plt.savefig(folder+'/eta_vs_cost_iter_'+name+'.png')
        plt.close()


def save_problem_data(other_params, q_init, q_fin, q_init_new, q_fin_new, q, q_new, q_pred,\
                      comp_time_1, comp_time_2, comp_time_3, saved_cost, saved_eta, \
                      saved_ys_pred, etas, cost_out_1, cost_out_2, folder=None):
    save_dict = {}
    save_dict['other_params'] = other_params
    save_dict['q_init'] = q_init
    save_dict['q_init_new'] = q_init_new
    save_dict['q_fin'] = q_fin
    save_dict['q_fin_new'] = q_fin_new
    save_dict['q'] = q
    save_dict['q_new'] = q_new
    save_dict['q_pred'] = q_pred
    save_dict['comp_time_1'] = comp_time_1
    save_dict['comp_time_2'] = comp_time_2
    save_dict['comp_time_3'] = comp_time_3
    save_dict['etas'] = etas
    save_dict['saved_cost'] = saved_cost
    save_dict['saved_eta'] = saved_eta
    save_dict['saved_ys_pred'] = saved_ys_pred
    save_dict['cost_out_1'] = cost_out_1
    save_dict['cost_out_2'] = cost_out_2

    if folder is None:
        print("Data is not saved")
    else:
        with open(folder+'/data.npy', 'wb') as f:
            np.save(f, save_dict)


def plot_joint_angles(q_new, q_pred, folder=None):
    q1_new, q2_new, q3_new, q4_new, q5_new, q6_new, q7_new = unstack_y(q_new)
    q1_pred, q2_pred, q3_pred, q4_pred, q5_pred, q6_pred, q7_pred = unstack_y(q_pred)
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(421)
    ax.plot(q1_new, '-k')
    ax.plot(q1_pred, '--k')
    ax.grid()
    ax.set_title('q1 angles')
    ax = fig.add_subplot(422)
    ax.plot(q2_new, '-k')
    ax.plot(q2_pred, '--k')
    ax.grid()
    ax.set_title('q2 angles')
    ax = fig.add_subplot(423)
    ax.plot(q3_new, '-k')
    ax.plot(q3_pred, '--k')
    ax.grid()
    ax.set_title('q3 angles')
    ax = fig.add_subplot(424)
    ax.plot(q4_new, '-k')
    ax.plot(q4_pred, '--k')
    ax.grid()
    ax.set_title('q4 angles')
    ax = fig.add_subplot(425)
    ax.plot(q5_new, '-k')
    ax.plot(q5_pred, '--k')
    ax.grid()
    ax.set_title('q5 angles')
    ax = fig.add_subplot(426)
    ax.plot(q6_new, '-k')
    ax.plot(q6_pred, '--k')
    ax.grid()
    ax.set_title('q6 angles')
    ax = fig.add_subplot(427)
    ax.plot(q7_new, '-k')
    ax.plot(q7_pred, '--k')
    ax.grid()
    ax.set_title('q7 angles')

    if folder is None:
        plt.show()
    else:
        plt.savefig(folder+'/joint_angles.png')
        plt.close()


def get_sin_cos(y_in):
    q_1, q_2, q_3, q_4, q_5, q_6, q_7 = unstack_y(y_in)
    sq1 = np.sin(q_1)
    sq2 = np.sin(q_2)
    sq3 = np.sin(q_3)
    sq4 = np.sin(q_4)
    sq5 = np.sin(q_5)
    sq6 = np.sin(q_6)
    sq7 = np.sin(q_7)
    cq1 = np.cos(q_1)
    cq2 = np.cos(q_2)
    cq3 = np.cos(q_3)
    cq4 = np.cos(q_4)
    cq5 = np.cos(q_5)
    cq6 = np.cos(q_6)
    cq7 = np.cos(q_7)
    return sq1, sq2, sq3, sq4, sq5, sq6, sq7, cq1, cq2, cq3, cq4, cq5, cq6, cq7


def get_roll_pitch(y_in):
    sq1, sq2, sq3, sq4, sq5, sq6, sq7, cq1, cq2, cq3, cq4, cq5, cq6, cq7 = get_sin_cos(y_in)
    r_32 = -cq7*(cq5*sq2*sq3 - sq5*(cq2*sq4 - cq3*cq4*sq2)) - sq7*(cq6*(cq5*(cq2*sq4 - cq3*cq4*sq2) + sq2*sq3*sq5) + sq6*(cq2*cq4 + cq3*sq2*sq4))
    r_33 = -cq6*(cq2*cq4 + cq3*sq2*sq4) + sq6*(cq5*(cq2*sq4 - cq3*cq4*sq2) + sq2*sq3*sq5)
    r_31 = cq7*(cq6*(cq5*(cq2*sq4 - cq3*cq4*sq2) + sq2*sq3*sq5) + sq6*(cq2*cq4 + cq3*sq2*sq4)) - sq7*(cq5*sq2*sq3 - sq5*(cq2*sq4 - cq3*cq4*sq2))
    return np.arctan2(r_32, r_33), -np.arcsin(r_31)


def get_sin_cos_jax(y_in):
    q_1, q_2, q_3, q_4, q_5, q_6, q_7 = unstack_y(y_in)
    sq1 = jnp.sin(q_1)
    sq2 = jnp.sin(q_2)
    sq3 = jnp.sin(q_3)
    sq4 = jnp.sin(q_4)
    sq5 = jnp.sin(q_5)
    sq6 = jnp.sin(q_6)
    sq7 = jnp.sin(q_7)
    cq1 = jnp.cos(q_1)
    cq2 = jnp.cos(q_2)
    cq3 = jnp.cos(q_3)
    cq4 = jnp.cos(q_4)
    cq5 = jnp.cos(q_5)
    cq6 = jnp.cos(q_6)
    cq7 = jnp.cos(q_7)
    return sq1, sq2, sq3, sq4, sq5, sq6, sq7, cq1, cq2, cq3, cq4, cq5, cq6, cq7


def get_roll_pitch_jax(y_in):
    sq1, sq2, sq3, sq4, sq5, sq6, sq7, cq1, cq2, cq3, cq4, cq5, cq6, cq7 = get_sin_cos_jax(y_in)
    r_32 = -cq7*(cq5*sq2*sq3 - sq5*(cq2*sq4 - cq3*cq4*sq2)) - sq7*(cq6*(cq5*(cq2*sq4 - cq3*cq4*sq2) + sq2*sq3*sq5) + sq6*(cq2*cq4 + cq3*sq2*sq4))
    r_33 = -cq6*(cq2*cq4 + cq3*sq2*sq4) + sq6*(cq5*(cq2*sq4 - cq3*cq4*sq2) + sq2*sq3*sq5)
    r_31 = cq7*(cq6*(cq5*(cq2*sq4 - cq3*cq4*sq2) + sq2*sq3*sq5) + sq6*(cq2*cq4 + cq3*sq2*sq4)) - sq7*(cq5*sq2*sq3 - sq5*(cq2*sq4 - cq3*cq4*sq2))

    return jnp.arctan2(r_32, r_33), -jnp.arcsin(r_31)


def hessian_yy(fun):
    return jit(jacfwd(jacrev(fun, argnums=0), argnums=0))


def hessian_xy(fun):
    return jit(jacfwd(jacrev(fun, argnums=0), argnums=1))


def clipping_angle_jax(y, other_params):
    return jnp.clip(y, other_params['q_min_traj'], other_params['q_max_traj'])


def solver_solution(x_guess, params, other_params, cost_fun):
    opts = {'maxiter': 600}
    q_min_traj = other_params['q_min_traj']
    q_max_traj = other_params['q_max_traj']
    bnds = Bounds(q_min_traj, q_max_traj)

    # autograd gradient
    cost_grad = grad(cost_fun, 0)
    start_time = time.time()
    res = minimize(cost_fun, x_guess, method='SLSQP', jac=cost_grad, bounds=bnds, options=opts, args=(params, other_params))
    total_time = time.time()-start_time
    return res.x, total_time, res.message


def save_to_dict(counter, cost, eta, ys_pred, saved_cost, saved_eta, saved_ys_pred):
    saved_cost[counter] = cost
    saved_eta[counter] = eta
    saved_ys_pred[counter] = ys_pred


def argmin_solution(ys, xs, xs_new, other_params, etas, niters, cost_fun_jax, vmap_batched_update_ys, F_YY_fn, F_XY_fn, new_fn, stack_x, folder=None):
    saved_cost = {}
    saved_eta = {}
    saved_ys_pred = {}

    xs = jnp.array(xs)
    xs_new = jnp.array(xs_new)
    param_diff = (xs_new-xs).reshape(-1, 1)

    start_time = time.time()
    ys_pred = jnp.array(ys.copy())
    prev_cost = cost_fun_jax(ys_pred, xs_new, other_params)
    curr_cost = prev_cost
    save_to_dict(0, curr_cost, -1, ys_pred, saved_cost, saved_eta, saved_ys_pred)

    for i in range(niters):
        prev_ys_pred = ys_pred
        F_YY = F_YY_fn(ys_pred, xs, other_params)
        F_XY = F_XY_fn(ys_pred, xs, other_params)
        dgx = jnp.linalg.solve(F_YY, -F_XY)
        dy = jnp.matmul(dgx, param_diff).reshape(ys.shape[0])
        ys_pred_vec, curr_cost_vec = vmap_batched_update_ys(ys_pred, dy, xs_new, other_params, jnp.array(etas))
        print(curr_cost_vec)
        index = jnp.argmin(curr_cost_vec)
        ys_pred = ys_pred_vec[index, :].copy()
        curr_cost = curr_cost_vec[index]
        save_to_dict(i+1, curr_cost_vec, etas[index], ys_pred, saved_cost, saved_eta, saved_ys_pred)
        print("selected eta : ", etas[index], "  located at index : ", index)
        if(curr_cost < prev_cost):
            prev_cost = curr_cost
            print(curr_cost)
        else:
            ys_pred = prev_ys_pred
            print("Final Cost : ", prev_cost)
            print("Iterations : ", i)
            break

        xs = stack_x(*(new_fn(ys_pred, other_params)))
        param_diff = (xs_new-xs).reshape(-1, 1)
        print("-"*50)

    total_time = time.time()-start_time
    return np.array(ys_pred), total_time, saved_cost, saved_eta, saved_ys_pred


def get_real_guess(data, trajcount, i):
    joints = data[trajcount]['traj'][i]['jointangles']
    joints = joints.T.flatten()
    return joints


def input_data_handle(i):
    if(i == 0):
        return "../dataset/mid_point_dataset.npy"
    elif(i == 1):
        return "../dataset/dataset_small.npy"
    elif(i == 2):
        return "../dataset/dataset_medium.npy"
    elif(i == 3):
        return "../dataset/dataset_large.npy"
    elif(i == 4):
        return "../dataset/dataset_xlarge.npy"
    else:
        print("Provide a valid input file")
        raise ValueError
