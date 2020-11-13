import sys
import os
sys.path.insert(0, "../")
import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import brewer2mpl
import matplotlib as mpl
import matplotlib.ticker as ticker

bmap = brewer2mpl.get_map('Dark2', 'qualitative', 7)
colors = bmap.mpl_colors

params = {
    'axes.labelsize': 20,
    'font.size': 20,
    'legend.fontsize': 20,
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'text.usetex': False,
    'font.family': 'serif',
    'axes.prop_cycle': mpl.cycler(color=colors)
}
mpl.rcParams.update(params)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

EXPERIMENT_NAME = "mid_point"
OUTPUT_FOLDER = "../images/"+EXPERIMENT_NAME+"_perturbation/"

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
    return f_orient_cost, f_smoothness_boundary, f_smoothness_vel, f_smoothness_acc, f_smoothness_jerk, f_pos_cost, f_pos_mid


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
    q_min = np.array([-165.0, -100.0, -165.0, -165.0, -165.0, -1.0, -165.0  ])*np.pi/180
    q_max = np.array([ 165.0,   101.0,  165.0,  1.0,    165.0, 214.0, 165.0  ])*np.pi/180
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


def get_cost_diff(row):
    return abs(row['actual_cost'] - row['best_cost'])


def get_pos_cost_diff(row):
    return abs(row['actual_f_boundary'] - row['best_f_boundary'])


def get_orient_cost_diff(row):
    return abs(row['actual_f_orient'] - row['best_f_orient'])


def get_vel_smoothness_cost_diff(row):
    return abs(row['actual_f_vel'] - row['best_f_vel'])


def get_pos_cost_diff_new(row):
    return abs(row['actual_f_pos_cost'] - row['best_f_pos_cost'])


def get_mid_cost_diff(row):
    return abs(row['actual_f_mid_cost'] - row['best_f_mid_cost'])


def get_accel_smoothness_cost_diff(row):
    return abs(row['actual_f_acc'] - row['best_f_acc'])


def get_jerk_smoothness_cost_diff(row):
    return abs(row['actual_f_jerk'] - row['best_f_jerk'])


def get_mid_point(y_in, mid_index):
    x, y, z, _, _, _, _ = utils.fk_franka_traj(y_in)
    return x[mid_index], y[mid_index], z[mid_index]


def main(perturb, save=False):
    all_list = []
    for trajcount in range(6):
        for i in range(40):
            try:

                folder = '../output/'+EXPERIMENT_NAME+'_perturbation/'+perturb+'/traj'+str(trajcount)+'/'+str(i) 

                data = np.load(folder+"/data.npy", allow_pickle=True)
                data = data.item()

                q_init = data['q_init']
                q_fin = data['q_fin']

                x_fin, y_fin, z_fin, _, _, _, _ = utils.fk_franka_traj(q_fin)
                x_fin = x_fin[0]
                y_fin = y_fin[0]
                z_fin = z_fin[0]
                other_params = get_other_params(q_init, x_fin, y_fin, z_fin)
        
                x_mid = data['x_mid']
                y_mid = data['y_mid']
                z_mid = data['z_mid']
                x_mid_new = data['x_mid_new']
                y_mid_new = data['y_mid_new']
                z_mid_new = data['z_mid_new']


                params = stack_x(x_mid_new, y_mid_new, z_mid_new)

                q = data['q']
                q_new = data['q_new']
                q_pred = data['q_pred']
                f_orient_cost, f_smoothness_boundary, f_smoothness_vel, f_smoothness_acc, f_smoothness_jerk, f_pos_cost, f_mid_cost = cost_fun(q_pred, params, other_params)
                f_orient_cost_new, f_smoothness_boundary_new, f_smoothness_vel_new, f_smoothness_acc_new, f_smoothness_jerk_new, f_pos_cost_new, f_mid_cost_new = cost_fun(q_new, params, other_params)

                param_des = params.copy()  
                xmid, ymid, zmid = get_mid_point(q_new, other_params['mid_index'])
                param_new = stack_x(xmid,ymid,zmid)
                xmid, ymid, zmid = get_mid_point(q_pred, other_params['mid_index'])
                param_pred = stack_x(xmid,ymid,zmid)
                l_inf_new = max(abs(param_new-param_des))
                l_inf_pred = max(abs(param_pred-param_des))
                ratio_metric = l_inf_pred/l_inf_new

                roll_pred, pitch_pred = utils.get_roll_pitch(q_pred)
                roll_new, pitch_new = utils.get_roll_pitch(q_new)

                all_list.append([i, data['cost_3'], data['cost_2'], np.linalg.norm(q_new-q_pred), \
                        f_smoothness_boundary, f_smoothness_boundary_new, f_orient_cost, f_orient_cost_new, \
                        f_smoothness_vel, f_smoothness_vel_new, f_pos_cost, f_pos_cost_new, \
                        f_mid_cost, f_mid_cost_new, l_inf_pred, l_inf_new, ratio_metric, max(abs(roll_new-roll_pred)), max(abs(pitch_new-pitch_pred))])


            except:
                pass
                

    column_names = ["traj_num", "best_cost", "actual_cost", "best_norm", "best_f_boundary", "actual_f_boundary", "best_f_orient", "actual_f_orient", "best_f_vel", "actual_f_vel", "best_f_pos_cost", "actual_f_pos_cost", "best_f_mid_cost", "actual_f_mid_cost", "param_res_best", "param_res_actual", "ratio_metric", "roll_abs_diff", "pitch_abs_diff"]

    df = pd.DataFrame(all_list, columns=column_names)

    df['cost_diff'] = df.apply(get_cost_diff, axis=1)
    df['boundary_diff'] = df.apply(get_pos_cost_diff, axis=1)
    df['orient_diff'] = df.apply(get_orient_cost_diff, axis=1)
    df['vel_diff'] = df.apply(get_vel_smoothness_cost_diff, axis=1)
    df['pos_diff'] = df.apply(get_pos_cost_diff_new, axis=1)
    df['mid_diff'] = df.apply(get_mid_cost_diff, axis=1)

    if(save):
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)

        # a = df['roll_abs_diff'].to_numpy()
        # print("Roll : ", np.count_nonzero(a<0.1), " Total: ", a.shape)
        # b = df['pitch_abs_diff'].to_numpy()
        # print("Pitch : ", np.count_nonzero(b<0.1), " Total: ", b.shape)
        # c = df['vel_diff'].to_numpy()
        # print(np.mean(c))
        # print("Vel cost : ", np.count_nonzero(c<0.01348), " Total: ", c.shape)
        # e1 = df['ratio_metric'].to_numpy()
        # print("ratio metric : ", np.count_nonzero(e1<1.2), " Total: ", e1.shape)
        # v1 = df['actual_f_vel'].to_numpy()
        # v2 = df['best_f_vel'].to_numpy()
        # v3 = np.concatenate((v1,v2))
        # print("Mean cost : ", np.mean(v3), " no of trajs: ", 2*a.shape[0])
        # print(v3.shape)

        
        # roll_abs_diff
        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
        _ = axs.hist(df['roll_abs_diff'].tolist(), bins=10, color=colors[0])
        _ = axs.grid(ls = '-.', lw = 1)
        _ = axs.set_xlabel('$L_{\infty}$ of Roll Difference (rad)')
        _ = axs.set_ylabel('No. of Trajectories')
        # axs.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
        # _ = axs.set_xlim([-0.25, 3.0])
        # _ = axs.set_ylim([0, 130])
        # _ = axs.set_yticks(np.arange(0, 61, 10))
        plt.savefig(OUTPUT_FOLDER+'roll_cost.pdf', bbox_inches='tight', dpi=200)
        plt.close()

        # pitch_abs_diff
        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
        _ = axs.hist(df['pitch_abs_diff'].tolist(), bins=10, color=colors[0])
        _ = axs.grid(ls = '-.', lw = 1)
        _ = axs.set_xlabel('$L_{\infty}$ of Pitch Difference (rad)')
        _ = axs.set_ylabel('No. of Trajectories')
        # axs.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
        # _ = axs.set_xlim([-0.25, 3.0])
        # _ = axs.set_ylim([0, 110])
        # _ = axs.set_yticks(np.arange(0, 61, 10))
        plt.savefig(OUTPUT_FOLDER+'pitch_cost.pdf', bbox_inches='tight', dpi=200)
        plt.close()

        # velocity_diff
        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
        _ = axs.hist(df['vel_diff'].tolist(), bins=10, color=colors[0])
        _ = axs.grid(ls = '-.', lw = 1)
        _ = axs.set_xlabel('Smoothness Cost Difference')
        _ = axs.set_ylabel('No. of Trajectories')
        # _ = axs.set_xticks(np.arange(0.0, 0.004, 0.001))
        # axs.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
        # _ = axs.set_ylim([0, 150])
        # _ = axs.set_yticks(np.arange(0, 61, 10))
        plt.savefig(OUTPUT_FOLDER+'vel_cost.pdf', bbox_inches='tight', dpi=200)
        plt.close()

        # boundary_diff
        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
        _ = axs.hist(df['ratio_metric'].tolist(), bins=10, color=colors[0])
        # _ = axs.hist(df['param_res_actual'].tolist(), bins=10, color=colors[1], alpha=0.4, label='Solver')
        _ = axs.grid(ls = '-.', lw = 1)
        _ = axs.set_xlabel('Residual Ratio of Via-Point. ($L_{\infty}$)')
        _ = axs.set_ylabel('No. of Trajectories')
        # _ = axs.set_xticks(np.arange(0.0, 0.02, 0.005))
        # axs.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
        # _ = axs.set_ylim([0, 150])
        # _ = axs.set_yticks(np.arange(0, 61, 10))
        plt.savefig(OUTPUT_FOLDER+'param_cost.pdf', bbox_inches='tight', dpi=200)
        plt.close()


    return df



if __name__ == '__main__':
    if(int(sys.argv[1]) == -1):
        df1 = main('1')
        df2 = main('2')
        df3 = main('3')
        df4 = main('4')

        x = ['Small', 'Medium', 'Large', 'XLarge']

        r1 = df1['roll_abs_diff'].tolist()
        r2 = df2['roll_abs_diff'].tolist()
        r3 = df3['roll_abs_diff'].tolist()
        r4 = df4['roll_abs_diff'].tolist()

        roll_new_cost_q1_list = [np.quantile(r1, 0.25), np.quantile(r2, 0.25), np.quantile(r3, 0.25), np.quantile(r4, 0.25)]
        roll_new_cost_q2_list = [np.quantile(r1, 0.50), np.quantile(r2, 0.50), np.quantile(r3, 0.50), np.quantile(r4, 0.50)]
        roll_new_cost_q3_list = [np.quantile(r1, 0.75), np.quantile(r2, 0.75), np.quantile(r3, 0.75), np.quantile(r4, 0.75)]

        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
        _ = axs.plot(x, roll_new_cost_q2_list, linewidth=2, linestyle='-')
        _ = axs.scatter(x, roll_new_cost_q2_list)
        # _ = axs.set_ylim([-0.005, 0.045])
        # _ = axs.set_yticks(np.arange(-0.005, 0.045, 0.005))
        _ = axs.grid(ls = '-.', lw = 1)
        _ = axs.fill_between(x, roll_new_cost_q1_list, roll_new_cost_q3_list, alpha=0.25, linewidth=0, color=colors[1])
        _ = axs.set_xlabel('Perturbation Ranges')
        _ = axs.set_ylabel('$L_{\infty}$ of Roll Difference (rad)')

        # Legend labels
        _ = axs.plot([], [], ' ', label="Small:     0.0 - 0.1m")
        _ = axs.plot([], [], ' ', label="Medium: 0.1 - 0.2m")
        _ = axs.plot([], [], ' ', label="Large:     0.2 - 0.3m")
        _ = axs.plot([], [], ' ', label="X-Large:  0.3 - 0.4m")

        legend = axs.legend(handlelength=-0.5, loc=2)
        frame = legend.get_frame()
        frame.set_facecolor('0.9')
        frame.set_edgecolor('0.9')
        plt.savefig(OUTPUT_FOLDER+'roll_all.pdf', bbox_inches='tight', dpi=200)
        plt.close()

        p1 = df1['pitch_abs_diff'].tolist()
        p2 = df2['pitch_abs_diff'].tolist()
        p3 = df3['pitch_abs_diff'].tolist()
        p4 = df4['pitch_abs_diff'].tolist()

        pitch_new_cost_q1_list = [np.quantile(p1, 0.25), np.quantile(p2, 0.25), np.quantile(p3, 0.25), np.quantile(p4, 0.25)]
        pitch_new_cost_q2_list = [np.quantile(p1, 0.50), np.quantile(p2, 0.50), np.quantile(p3, 0.50), np.quantile(p4, 0.50)]
        pitch_new_cost_q3_list = [np.quantile(p1, 0.75), np.quantile(p2, 0.75), np.quantile(p3, 0.75), np.quantile(p4, 0.75)]

        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
        _ = axs.plot(x, pitch_new_cost_q2_list, linewidth=2, linestyle='-')
        _ = axs.scatter(x, pitch_new_cost_q2_list)
        # _ = axs.set_ylim([-0.005, 0.05])
        # _ = axs.set_yticks(np.arange(-0.005, 0.05, 0.005))
        _ = axs.grid(ls = '-.', lw = 1)
        _ = axs.fill_between(x, pitch_new_cost_q1_list, pitch_new_cost_q3_list, alpha=0.25, linewidth=0, color=colors[1])
        _ = axs.set_xlabel('Perturbation Ranges')
        _ = axs.set_ylabel('$L_{\infty}$ of Pitch Difference (rad)')

        # Legend labels
        _ = axs.plot([], [], ' ', label="Small:     0.0 - 0.1m")
        _ = axs.plot([], [], ' ', label="Medium: 0.1 - 0.2m")
        _ = axs.plot([], [], ' ', label="Large:     0.2 - 0.3m")
        _ = axs.plot([], [], ' ', label="X-Large:  0.3 - 0.4m")

        legend = axs.legend(handlelength=-0.5, loc=2)
        frame = legend.get_frame()
        frame.set_facecolor('0.9')
        frame.set_edgecolor('0.9')
        plt.savefig(OUTPUT_FOLDER+'pitch_all.pdf', bbox_inches='tight', dpi=200)
        plt.close()


        v1 = df1['vel_diff'].tolist()
        v2 = df2['vel_diff'].tolist()
        v3 = df3['vel_diff'].tolist()
        v4 = df4['vel_diff'].tolist()

        vel_new_cost_q1_list = [np.quantile(v1, 0.25), np.quantile(v2, 0.25), np.quantile(v3, 0.25), np.quantile(v4, 0.25)]
        vel_new_cost_q2_list = [np.quantile(v1, 0.50), np.quantile(v2, 0.50), np.quantile(v3, 0.50), np.quantile(v4, 0.50)]
        vel_new_cost_q3_list = [np.quantile(v1, 0.75), np.quantile(v2, 0.75), np.quantile(v3, 0.75), np.quantile(v4, 0.75)]

        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
        _ = axs.plot(x, vel_new_cost_q2_list, linewidth=2, linestyle='-')
        _ = axs.scatter(x, vel_new_cost_q2_list)
        # _ = axs.set_ylim([-0.0002, 0.0014])
        # _ = axs.set_yticks(np.arange(-0.0002, 0.0014, 0.0002))
        _ = axs.grid(ls = '-.', lw = 1)
        _ = axs.fill_between(x, vel_new_cost_q1_list, vel_new_cost_q3_list, alpha=0.25, linewidth=0, color=colors[1])
        _ = axs.set_xlabel('Perturbation Ranges')
        _ = axs.set_ylabel('Smoothness Cost Difference')
        # axs.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))

        # Legend labels
        _ = axs.plot([], [], ' ', label="Small:     0.0 - 0.1m")
        _ = axs.plot([], [], ' ', label="Medium: 0.1 - 0.2m")
        _ = axs.plot([], [], ' ', label="Large:     0.2 - 0.3m")
        _ = axs.plot([], [], ' ', label="X-Large:  0.3 - 0.4m")

        legend = axs.legend(handlelength=-0.5, loc=2)
        frame = legend.get_frame()
        frame.set_facecolor('0.9')
        frame.set_edgecolor('0.9')
        plt.savefig(OUTPUT_FOLDER+'vel_all.pdf', bbox_inches='tight', dpi=200)
        plt.close()

        p1 = df1['ratio_metric'].tolist()
        p2 = df2['ratio_metric'].tolist()
        p3 = df3['ratio_metric'].tolist()
        p4 = df4['ratio_metric'].tolist()

        p_new_cost_q1_list = [np.quantile(p1, 0.25), np.quantile(p2, 0.25), np.quantile(p3, 0.25), np.quantile(p4, 0.25)]
        p_new_cost_q2_list = [np.quantile(p1, 0.50), np.quantile(p2, 0.50), np.quantile(p3, 0.50), np.quantile(p4, 0.50)]
        p_new_cost_q3_list = [np.quantile(p1, 0.75), np.quantile(p2, 0.75), np.quantile(p3, 0.75), np.quantile(p4, 0.75)]

        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
        _ = axs.plot(x, p_new_cost_q2_list, linewidth=2, linestyle='-')
        _ = axs.scatter(x, p_new_cost_q2_list)
        _ = axs.set_ylim([0.9, 2.1])
        _ = axs.set_yticks(np.arange(0.9, 2.1, 0.2))
        _ = axs.grid(ls = '-.', lw = 1)
        _ = axs.fill_between(x, p_new_cost_q1_list, p_new_cost_q3_list, alpha=0.25, linewidth=0, color=colors[1])
        _ = axs.set_xlabel('Perturbation Ranges')
        _ = axs.set_ylabel('Residual Ratio of Via-Point. ($L_{\infty}$)')
        # axs.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))

        # Legend labels
        _ = axs.plot([], [], ' ', label="Small:     0.0 - 0.1m")
        _ = axs.plot([], [], ' ', label="Medium: 0.1 - 0.2m")
        _ = axs.plot([], [], ' ', label="Large:     0.2 - 0.3m")
        _ = axs.plot([], [], ' ', label="X-Large:  0.3 - 0.4m")

        legend = axs.legend(handlelength=-0.5, loc=2)
        frame = legend.get_frame()
        frame.set_facecolor('0.9')
        frame.set_edgecolor('0.9')
        plt.savefig(OUTPUT_FOLDER+'param_all.pdf', bbox_inches='tight', dpi=200)
        plt.close()


    else:
        main(str(sys.argv[1]), True)
