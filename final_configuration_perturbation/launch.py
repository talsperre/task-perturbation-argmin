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


EXPERIMENT_NAME = "final_configuration"
OUTPUT_FOLDER = "../output/"+EXPERIMENT_NAME+"_perturbation/"
SOLVER = "../saved_solutions/"+EXPERIMENT_NAME+"_perturbation/"


def stack_x(q_init, q_fin):
    return np.concatenate((q_init, q_fin))


def unstack_x(x):
    q_1_init = x[0]
    q_2_init = x[1]
    q_3_init = x[2]
    q_4_init = x[3]
    q_5_init = x[4]
    q_6_init = x[5]
    q_7_init = x[6]
    q_1_fin = x[7]
    q_2_fin = x[8]
    q_3_fin = x[9]
    q_4_fin = x[10]
    q_5_fin = x[11]
    q_6_fin = x[12]
    q_7_fin = x[13]
    return q_1_init, q_2_init, q_3_init, q_4_init, q_5_init, q_6_init, q_7_init, q_1_fin, q_2_fin, q_3_fin, q_4_fin, q_5_fin, q_6_fin, q_7_fin


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

    q_1, q_2, q_3, q_4, q_5, q_6, q_7 = utils.unstack_y(y_in)
    q_1_init, q_2_init, q_3_init, q_4_init, q_5_init, q_6_init, q_7_init, q_1_fin, q_2_fin, q_3_fin, q_4_fin, q_5_fin, q_6_fin, q_7_fin = unstack_x(x_in)
    roll, pitch = utils.get_roll_pitch(y_in)

    f_orient_cost = np.sum((roll-roll_des)**2 + (pitch-pitch_des)**2)

    f_smoothness_boundary = ((q_1[0]-q_1_init)**2+(q_1[-1]-q_1_fin)**2+(q_2[0]-q_2_init)**2+ \
                            (q_2[-1]-q_2_fin)**2+(q_3[0]-q_3_init)**2+(q_3[-1]-q_3_fin)**2+ \
                            (q_4[0]-q_4_init)**2+(q_4[-1]-q_4_fin)**2+(q_5[0]-q_5_init)**2+ \
                            (q_5[-1]-q_5_fin)**2+(q_6[0]-q_6_init)**2+(q_6[-1]-q_6_fin)**2+ \
                            (q_7[0]-q_7_init)**2+(q_7[-1]-q_7_fin)**2)
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
    cost = rho_orient*f_orient_cost+f_smoothness_cost
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

    q_1, q_2, q_3, q_4, q_5, q_6, q_7 = utils.unstack_y(y_in)
    q_1_init, q_2_init, q_3_init, q_4_init, q_5_init, q_6_init, q_7_init, q_1_fin, q_2_fin, q_3_fin, q_4_fin, q_5_fin, q_6_fin, q_7_fin = unstack_x(x_in)
    roll, pitch = utils.get_roll_pitch_jax(y_in)

    f_orient_cost = np.sum((roll-roll_des)**2 + (pitch-pitch_des)**2)

    f_smoothness_boundary = ((q_1[0]-q_1_init)**2+(q_1[-1]-q_1_fin)**2+(q_2[0]-q_2_init)**2+ \
                            (q_2[-1]-q_2_fin)**2+(q_3[0]-q_3_init)**2+(q_3[-1]-q_3_fin)**2+ \
                            (q_4[0]-q_4_init)**2+(q_4[-1]-q_4_fin)**2+(q_5[0]-q_5_init)**2+ \
                            (q_5[-1]-q_5_fin)**2+(q_6[0]-q_6_init)**2+(q_6[-1]-q_6_fin)**2+ \
                            (q_7[0]-q_7_init)**2+(q_7[-1]-q_7_fin)**2)
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
    cost = rho_orient*f_orient_cost+f_smoothness_cost
    return cost


F_YY_fn = utils.hessian_yy(cost_fun_jax)
F_XY_fn = utils.hessian_xy(cost_fun_jax)


def get_other_params():

    other_params = {}

    n = 50
    other_params['n'] = 50

    # weight's of cost function
    other_params['rho_vel'] = 100.0
    other_params['rho_acc'] = 1000.0
    other_params['rho_b'] = 1000.0
    other_params['rho_jerk'] = 10000.0
    other_params['rho_orient'] = 10.0

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

    return other_params


@jit
def get_qfin_jax(ys_pred, other_params):
    q_1, q_2, q_3, q_4, q_5, q_6, q_7 = utils.unstack_y(ys_pred)
    q_init = jnp.array([q_1[0], q_2[0], q_3[0], q_4[0], q_5[0], q_6[0], q_7[0]])
    q_fin = jnp.array([q_1[-1], q_2[-1], q_3[-1], q_4[-1], q_5[-1], q_6[-1], q_7[-1]])
    return q_init, q_fin


def update_ys(ys_pred, dy, xs_new, other_params, eta):
    ys_pred = ys_pred + eta*dy
    ys_pred = utils.clipping_angle_jax(ys_pred, other_params)
    curr_cost = cost_fun_jax(ys_pred, xs_new, other_params)
    return ys_pred, curr_cost


@jit
def vmap_batched_update_ys(ys_pred, dy, xs_new, other_params, eta_batched):
  return vmap(partial(update_ys, ys_pred, dy, xs_new, other_params))(eta_batched)


def save_problem_data(q_init, q_fin, q_init_new, q_fin_new, q, q_new, q_pred, \
        comp_time_1, comp_time_2, comp_time_3, saved_cost, saved_eta, saved_ys_pred, other_params, folder):

    save_dict = {}
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
    params = stack_x(q_init, q_fin)
    save_dict['cost_1'] = cost_fun(q, params, other_params)
    params = stack_x(q_init_new, q_fin_new)
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


def main(input_file, input_no):
    other_params = get_other_params()
    data = np.load(input_file, allow_pickle=True)
    data = data.item()
    counter = 5
    for trajcount in range(6):
        q_init = data[trajcount]['q_init']
        q_fin = data[trajcount]['q_fin'][0]
        x_guess = utils.get_real_guess(data, trajcount, 0)
        xs = stack_x(q_init, q_fin)
        q, comp_time_1, msg = utils.solver_solution(x_guess, xs, other_params, cost_fun)
        print("Initial problem solved")
        for i in range(1, 7):
            n = data[trajcount]['q_fin'][i].shape[0]
            for j in range(n):
                folder = OUTPUT_FOLDER+"/"+input_no+"/traj"+str(trajcount)+'/'+str((i-1)*counter+j) 
                if not os.path.exists(folder):
                    os.makedirs(folder)

                q_init_new = q_init.copy()
                q_fin_new = data[trajcount]['q_fin'][i][j, :]

                ys = q.copy()
                x_guess = ys.copy()
                xs_new = stack_x(q_init_new, q_fin_new)
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
                                                                          F_XY_fn, get_qfin_jax, stack_x, folder)
                print("Perturbed problem solved by argmin")
                print("Time taken : ", comp_time_3)

                save_problem_data(q_init, q_fin, q_init_new, q_fin_new, q, q_new, q_pred, \
                                  comp_time_1, comp_time_2, comp_time_3, saved_cost, saved_eta, saved_ys_pred, other_params, folder)

                # plotting
                utils.plot_trajectory(q, q_new, q_pred, q_init, q_fin, q_init_new, q_fin_new, folder)
                utils.plot_end_effector_angles(q_new, q_pred, folder)
                utils.plot_joint_angles(q_new, q_pred, folder)
                print("-"*50)
            print("*"*50)
        print("="*80)
    print('done')


if __name__ == '__main__':
    if(len(sys.argv) > 1):
        input_file = utils.input_data_handle(int(sys.argv[1]))
        print("Using : ", input_file)
    else:
        print("Provide an input file")
        raise ValueError
    main(input_file, str(sys.argv[1]))
