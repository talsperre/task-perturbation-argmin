import os
import pdb
import sys
sys.path.insert(0, "../")
import utils

import autograd.numpy as np
from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit, vmap
from functools import partial


EXPERIMENT_NAME = "mid_point"
OUTPUT_FOLDER = "../output/"+EXPERIMENT_NAME+"_perturbation/"
SOLVER = "../saved_solutions/"+EXPERIMENT_NAME+"_perturbation/"


def stack_x(x_mid, y_mid, z_mid):
    return np.array([x_mid, y_mid, z_mid])


def unstack_x(x):
    x_mid = x[0]
    y_mid = x[1]
    z_mid = x[2]
    return x_mid, y_mid, z_mid


def cost_fun(y_in, x_in, other_params):

    roll_des = other_params['roll_des']
    pitch_des = other_params['pitch_des']

    A_vel = other_params['A_vel']
    A_acc = other_params['A_acc']
    A_jerk = other_params['A_jerk']

    rho_vel = other_params['rho_vel']
    rho_acc = other_params['rho_acc']
    rho_jerk = other_params['rho_jerk']
    rho_b = other_params['rho_b']
    rho_orient = other_params['rho_orient']
    rho_pos = other_params['rho_pos']

    q_1_init = other_params['q_1_init']
    q_2_init = other_params['q_2_init']
    q_3_init = other_params['q_3_init']
    q_4_init = other_params['q_4_init']
    q_5_init = other_params['q_5_init']
    q_6_init = other_params['q_6_init']
    q_7_init = other_params['q_7_init']

    x_fin = other_params['x_fin']
    y_fin = other_params['y_fin']
    z_fin = other_params['z_fin']

    mid_index = other_params['mid_index']

    q_1, q_2, q_3, q_4, q_5, q_6, q_7 = utils.unstack_y(y_in)
    x_mid, y_mid, z_mid = unstack_x(x_in)
    x, y, z, tr, ax1, ax2, ax3 = utils.fk_franka_traj(y_in)
    roll, pitch = utils.get_roll_pitch(y_in)

    f_orient_cost = np.sum((roll-roll_des)**2 + (pitch-pitch_des)**2)

    f_pos_cost = (x[-1]-x_fin)**2+(y[-1]-y_fin)**2+(z[-1]-z_fin)**2

    f_pos_mid = (x[mid_index]-x_mid)**2+(y[mid_index]-y_mid)**2+(z[mid_index]-z_mid)**2

    f_smoothness_boundary = ((q_1[0]-q_1_init)**2+(q_2[0]-q_2_init)**2+ \
                            (q_3[0]-q_3_init)**2+ (q_4[0]-q_4_init)**2+ \
                            (q_5[0]-q_5_init)**2+ (q_6[0]-q_6_init)**2+\
                            (q_7[0]-q_7_init)**2)
    f_smoothness_vel = np.sum(np.dot(A_vel, q_1)**2)+np.sum(np.dot(A_vel, q_2)**2)+ \
                        np.sum(np.dot(A_vel, q_3)**2)+np.sum(np.dot(A_vel, q_4)**2)+ \
                        np.sum(np.dot(A_vel, q_5)**2)+np.sum(np.dot(A_vel, q_6)**2)+ \
                        np.sum(np.dot(A_vel, q_7)**2)
    f_smoothness_acc = np.sum(np.dot(A_acc, q_1)**2)+np.sum(np.dot(A_acc, q_2)**2)+ \
                        np.sum(np.dot(A_acc, q_3)**2)+np.sum(np.dot(A_acc, q_4)**2)+ \
                        np.sum(np.dot(A_acc, q_5)**2)+np.sum(np.dot(A_acc, q_6)**2)+ \
                        np.sum(np.dot(A_acc, q_7)**2)
    f_smoothness_jerk = np.sum(np.dot(A_jerk, q_1)**2)+np.sum(np.dot(A_jerk, q_2)**2)+ \
                        np.sum(np.dot(A_jerk, q_3)**2)+np.sum(np.dot(A_jerk, q_4)**2)+ \
                        np.sum(np.dot(A_jerk, q_5)**2)+np.sum(np.dot(A_jerk, q_6)**2)+ \
                        np.sum(np.dot(A_jerk, q_7)**2)

    f_smoothness_cost = rho_vel*f_smoothness_vel+rho_acc*f_smoothness_acc+rho_b*f_smoothness_boundary+rho_jerk*f_smoothness_jerk
    cost = rho_orient*f_orient_cost+f_smoothness_cost+rho_pos*(f_pos_cost+f_pos_mid)
    return cost


@jit
def cost_fun_jax(y_in, x_in, other_params):

    roll_des = other_params['roll_des']
    pitch_des = other_params['pitch_des']

    A_vel = other_params['A_vel']
    A_acc = other_params['A_acc']
    A_jerk = other_params['A_jerk']

    rho_vel = other_params['rho_vel']
    rho_acc = other_params['rho_acc']
    rho_jerk = other_params['rho_jerk']
    rho_b = other_params['rho_b']
    rho_orient = other_params['rho_orient']
    rho_pos = other_params['rho_pos']

    q_1_init = other_params['q_1_init']
    q_2_init = other_params['q_2_init']
    q_3_init = other_params['q_3_init']
    q_4_init = other_params['q_4_init']
    q_5_init = other_params['q_5_init']
    q_6_init = other_params['q_6_init']
    q_7_init = other_params['q_7_init']

    x_fin = other_params['x_fin']
    y_fin = other_params['y_fin']
    z_fin = other_params['z_fin']

    mid_index = other_params['mid_index']

    q_1, q_2, q_3, q_4, q_5, q_6, q_7 = utils.unstack_y(y_in)
    x_mid, y_mid, z_mid = unstack_x(x_in)
    x, y, z, tr, ax1, ax2, ax3 = utils.fk_franka_traj_jax(y_in)
    roll, pitch = utils.get_roll_pitch_jax(y_in)

    f_orient_cost = np.sum((roll-roll_des)**2 + (pitch-pitch_des)**2)

    f_pos_cost = (x[-1]-x_fin)**2+(y[-1]-y_fin)**2+(z[-1]-z_fin)**2

    f_pos_mid = (x[mid_index]-x_mid)**2+(y[mid_index]-y_mid)**2+(z[mid_index]-z_mid)**2

    f_smoothness_boundary = ((q_1[0]-q_1_init)**2+(q_2[0]-q_2_init)**2+ \
                            (q_3[0]-q_3_init)**2+ (q_4[0]-q_4_init)**2+ \
                            (q_5[0]-q_5_init)**2+ (q_6[0]-q_6_init)**2+\
                            (q_7[0]-q_7_init)**2)

    f_smoothness_vel = jnp.sum(jnp.dot(A_vel, q_1)**2)+jnp.sum(jnp.dot(A_vel, q_2)**2)+ \
                        jnp.sum(jnp.dot(A_vel, q_3)**2)+jnp.sum(jnp.dot(A_vel, q_4)**2)+ \
                        jnp.sum(jnp.dot(A_vel, q_5)**2)+jnp.sum(jnp.dot(A_vel, q_6)**2)+ \
                        jnp.sum(jnp.dot(A_vel, q_7)**2)
    f_smoothness_acc = jnp.sum(jnp.dot(A_acc, q_1)**2)+jnp.sum(jnp.dot(A_acc, q_2)**2)+ \
                        jnp.sum(jnp.dot(A_acc, q_3)**2)+jnp.sum(jnp.dot(A_acc, q_4)**2)+ \
                        jnp.sum(jnp.dot(A_acc, q_5)**2)+jnp.sum(jnp.dot(A_acc, q_6)**2)+ \
                        jnp.sum(jnp.dot(A_acc, q_7)**2)
    f_smoothness_jerk = jnp.sum(jnp.dot(A_jerk, q_1)**2)+jnp.sum(jnp.dot(A_jerk, q_2)**2)+ \
                        jnp.sum(jnp.dot(A_jerk, q_3)**2)+jnp.sum(jnp.dot(A_jerk, q_4)**2)+ \
                        jnp.sum(jnp.dot(A_jerk, q_5)**2)+jnp.sum(jnp.dot(A_jerk, q_6)**2)+ \
                        jnp.sum(jnp.dot(A_jerk, q_7)**2)

    f_smoothness_cost = rho_vel*f_smoothness_vel+rho_acc*f_smoothness_acc+rho_b*f_smoothness_boundary+rho_jerk*f_smoothness_jerk
    cost = rho_orient*f_orient_cost+f_smoothness_cost+rho_pos*(f_pos_cost+f_pos_mid)
    return cost


F_YY_fn = utils.hessian_yy(cost_fun_jax)
F_XY_fn = utils.hessian_xy(cost_fun_jax)


def get_other_params(q_init, x_fin, y_fin, z_fin):

    other_params = {}

    n = 50
    other_params['n'] = 50

    # weight's of cost function
    other_params['rho_vel'] = 100.0
    other_params['rho_acc'] = 1000.0
    other_params['rho_b'] = 1000.0
    other_params['rho_jerk'] = 10000.0
    other_params['rho_orient'] = 1000.0
    other_params['rho_pos'] = 1000.0

    # joint limits of franka manipulator
    q_min = np.array([-165.0, -100.0, -165.0, -165.0, -165.0, -1.0, -165.0])*np.pi/180
    q_max = np.array([165.0,   101.0,  165.0,  1.0,    165.0, 214.0, 165.0])*np.pi/180
    other_params['q_min_traj'] = np.hstack((q_min[0]*np.ones(n), q_min[1]*np.ones(n), q_min[2]*np.ones(n),\
                                            q_min[3]*np.ones(n), q_min[4]*np.ones(n), q_min[5]*np.ones(n), \
                                            q_min[6]*np.ones(n)))
    other_params['q_max_traj'] = np.hstack((q_max[0]*np.ones(n), q_max[1]*np.ones(n), q_max[2]*np.ones(n), \
                                            q_max[3]*np.ones(n), q_max[4]*np.ones(n), q_max[5]*np.ones(n), \
                                            q_max[6]*np.ones(n)))

    # first, second and third order difference matrices
    A = np.identity(n)
    other_params['A_vel'] = np.diff(A, axis=0)
    other_params['A_acc'] = np.diff(other_params['A_vel'], axis=0)
    other_params['A_jerk'] = np.diff(other_params['A_acc'], axis=0)

    # desired axis-angle for end effector
    roll_des = -87.2*np.pi/180
    pitch_des = -41.0*np.pi/180

    other_params['roll_des'] = roll_des
    other_params['pitch_des'] = pitch_des

    # mid point index
    other_params['mid_index'] = 25

    other_params['q_1_init'] = q_init[0]
    other_params['q_2_init'] = q_init[1]
    other_params['q_3_init'] = q_init[2]
    other_params['q_4_init'] = q_init[3]
    other_params['q_5_init'] = q_init[4]
    other_params['q_6_init'] = q_init[5]
    other_params['q_7_init'] = q_init[6]

    other_params['x_fin'] = x_fin
    other_params['y_fin'] = y_fin
    other_params['z_fin'] = z_fin

    return other_params


def update_ys(ys_pred, dy, xs_new, other_params, eta):
    ys_pred = ys_pred + eta*dy
    ys_pred = utils.clipping_angle_jax(ys_pred, other_params)
    curr_cost = cost_fun_jax(ys_pred, xs_new, other_params)
    return ys_pred, curr_cost


@jit
def vmap_batched_update_ys(ys_pred, dy, xs_new, other_params, eta_batched):
  return vmap(partial(update_ys, ys_pred, dy, xs_new, other_params))(eta_batched)


def save_problem_data(q_init, q_fin, x_mid, y_mid, z_mid, x_mid_new, y_mid_new, z_mid_new, q, q_new, q_pred, \
        comp_time_1, comp_time_2, comp_time_3, saved_cost, saved_eta, saved_ys_pred, other_params, folder):

    save_dict = {}
    save_dict['q_init'] = q_init
    save_dict['q_fin'] = q_fin
    save_dict['x_mid'] = x_mid
    save_dict['y_mid'] = y_mid
    save_dict['z_mid'] = z_mid
    save_dict['x_mid_new'] = x_mid_new
    save_dict['y_mid_new'] = y_mid_new
    save_dict['z_mid_new'] = z_mid_new
    save_dict['q'] = q
    save_dict['q_new'] = q_new
    save_dict['q_pred'] = q_pred
    save_dict['comp_time_1'] = comp_time_1
    save_dict['comp_time_2'] = comp_time_2
    save_dict['comp_time_3'] = comp_time_3
    params = stack_x(x_mid, y_mid, z_mid)
    save_dict['cost_1'] = cost_fun(q, params, other_params)
    params = stack_x(x_mid_new, y_mid_new, z_mid_new)
    save_dict['cost_2'] = cost_fun(q_new, params, other_params)
    save_dict['cost_3'] = cost_fun(q_pred, params, other_params)
    save_dict['saved_cost'] = saved_cost
    save_dict['saved_eta'] = saved_eta
    save_dict['saved_ys_pred'] = saved_ys_pred

    if folder is None:
        print("Data is not saved")
    else:
        with open(folder+'/data.npy', 'wb') as f:
            np.save(f, save_dict)


def get_mid_point(y_in, other_params):
    mid_index = other_params['mid_index']
    x, y, z, _, _, _, _ = utils.fk_franka_traj(y_in)
    return x[mid_index], y[mid_index], z[mid_index]


@jit
def get_mid_point_jax(y_in, other_params):
    mid_index = other_params['mid_index']
    x, y, z, _, _, _, _ = utils.fk_franka_traj_jax(y_in)
    return x[mid_index], y[mid_index], z[mid_index]


def select_min_max_perturbation(input_no):
    return (input_no-1)*0.1, input_no*0.1


def main(input_file, input_no):
    data = np.load(input_file, allow_pickle=True)
    data = data.item()
    min_perturb, max_perturb = select_min_max_perturbation(int(input_no))
    counter = 5
    np.random.seed(322)
    for trajcount in range(6):
        for i in range(7):
            q_init = data[trajcount]['q_init'][i]
            q_fin = data[trajcount]['q_fin'][i]

            x_fin, y_fin, z_fin, _, _, _, _ = utils.fk_franka_traj(q_fin)
            x_fin = x_fin[0]
            y_fin = y_fin[0]
            z_fin = z_fin[0]
            other_params = get_other_params(q_init, x_fin, y_fin, z_fin)

            x_guess = utils.get_real_guess(data, trajcount, i)
            x_mid, y_mid, z_mid = get_mid_point(x_guess, other_params)
            xs = stack_x(x_mid, y_mid, z_mid)
            q, comp_time_1, msg = utils.solver_solution(x_guess, xs, other_params, cost_fun)
            print("Initial problem solved")
            for j in range(counter):
                folder = OUTPUT_FOLDER+"/"+input_no+"/traj"+str(trajcount)+'/'+str(i*counter+j) 
                if not os.path.exists(folder):
                    os.makedirs(folder)

                ys = q.copy()
                dxs = np.random.uniform(low=min_perturb, high=max_perturb, size=(3,))
                x_mid_new = x_mid+dxs[0]
                y_mid_new = y_mid+dxs[1]
                z_mid_new = z_mid+dxs[2]
                xs_new = stack_x(x_mid_new, y_mid_new, z_mid_new)

                x_guess = ys.copy()
                q_new, comp_time_2, msg = utils.solver_solution(x_guess, xs_new, other_params, cost_fun)
                print("Perturbed problem solved by solver")
                solver_sol = {}
                solver_sol['q'] = q
                solver_sol['q_new'] = q_new
                solver_sol['comp_time_1'] = comp_time_1
                solver_sol['comp_time_2'] = comp_time_2
                sfolder = SOLVER+"/"+input_no+"/traj"+str(trajcount)+'/'+str((i-1)*counter+j)
                if not os.path.exists(sfolder):
                    os.makedirs(sfolder)
                with open(sfolder+'/data.npy', 'wb') as f:
                        np.save(f, solver_sol)

                print("Minimal cost from solver intial ", cost_fun(q, xs, other_params))
                print("Minimal cost from solver perturbed ", cost_fun(q_new, xs_new, other_params))
                etas = np.arange(0.05, 1.05, 0.05)
                niters = 30
                q_pred, comp_time_3, saved_cost, saved_eta, saved_ys_pred = utils.argmin_solution(ys, xs, xs_new, other_params, \
                                                                          etas, niters, cost_fun_jax, vmap_batched_update_ys, F_YY_fn, \
                                                                          F_XY_fn, get_mid_point_jax, stack_x, folder)
                print("Perturbed problem solved by argmin")
                print("Time taken : ", comp_time_3)

                save_problem_data(q_init, q_fin, x_mid, y_mid, z_mid, x_mid_new, y_mid_new, z_mid_new, q, q_new, q_pred, \
                                  comp_time_1, comp_time_2, comp_time_3, saved_cost, saved_eta, saved_ys_pred, other_params, folder)

                # plotting
                # folder = None # shows interactive plots
                utils.plot_trajectory_mid(q, q_new, q_pred, q_init, q_fin, x_mid, y_mid, z_mid, x_mid_new, y_mid_new, z_mid_new, folder)
                utils.plot_end_effector_angles(q_new, q_pred, folder)
                utils.plot_joint_angles(q_new, q_pred, folder)
                print("-"*50)
            print("*"*50)
        print("="*80)
    print('done')


if __name__ == '__main__':
    if(len(sys.argv) > 1):
        input_file = utils.input_data_handle(0)
        print("Using : ", input_file)
    else:
        print("Provide an input file")
        raise ValueError
    main(input_file, str(sys.argv[1]))
