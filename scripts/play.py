import isaacgym

assert isaacgym
import torch
import numpy as np

import glob
import pickle as pkl

from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

from tqdm import tqdm

joy_cmd = [0.0, 0.0, 0.0]
stop=False

def joy_callback(joy_msg):
    global joy_cmd
    global stop
    joy_cmd[0] =  3*joy_msg.axes[1]
    joy_cmd[1] =  0.6*joy_msg.axes[0]
    joy_cmd[2] =  3*joy_msg.axes[3]  # 横向操作

    if(joy_cmd[0]>3 ):
        joy_cmd[0]=3
    if(joy_cmd[0]<-3):
        joy_cmd[0]=-3
    if(joy_cmd[1]>0.6):
        joy_cmd[1]=0.6
    if (joy_cmd[1] < -0.6):
        joy_cmd[1] = -0.6
    if (joy_cmd[2] > 3):
        joy_cmd[2] = 3
    if (joy_cmd[2] < -3 ):
        joy_cmd[2] = -3

    if(joy_msg.buttons[1]):
        stop=True


def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    import os
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


def load_env(label, headless=False):
    dirs = glob.glob(f"../runs/{label}/*")
    logdir = sorted(dirs)[-1]

    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"

    Cfg.viewer.pos=[10,10,6]
    Cfg.viewer.lookat=[11,5,3]

    Cfg.terrain.selected = False
    Cfg.terrain.mesh_type = 'trimesh'
    Cfg.terrain.selected_terrain_type = "pyramid_stairs"
    Cfg.terrain.terrain_kwargs = {
        'random_uniform':
            {'min_height': -0.082,
             'max_height': 0.082,
             'step': 0.005,
             'downsampled_scale': 0.2
             },
        'pyramid_sloped':
            {'slope': -0.45,
             'platform_size': 3.
             },
        'pyramid_stairs':
            {'step_width': 0.4,
             'step_height': 0.1,
             'platform_size': 1.
             },
        'discrete_obstacles':
            {
                'max_height': 0.05,
                'min_size': 1.,
                'max_size': 2.,
                'num_rects': 20,
                'platform_size': 3.
            }
    }  # Dict of arguments for selected terrain

    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=False, cfg=Cfg)
    env = HistoryWrapper(env)

    # load policy
    from ml_logger import logger
    from go1_gym_learn.ppo_cse.actor_critic import ActorCritic

    policy = load_policy(logdir)

    return env, policy


def play_go1(headless=True):
    from ml_logger import logger

    from pathlib import Path
    from go1_gym import MINI_GYM_ROOT_DIR
    import glob
    import os

    # import rospy
    # from sensor_msgs.msg import Joy
    #
    # rospy.init_node('play')
    # rospy.Subscriber('/joy', Joy, joy_callback, queue_size=10)

    # label = "gait-conditioned-agility/2024-09-08/train"
    label = "gait-conditioned-agility/pretrain-v0/train"

    env, policy = load_env(label, headless=headless)

    num_eval_steps = 20000
    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}

    body_height_cmd = 0
    step_frequency_cmd = 4.0
    gait = torch.tensor(gaits["trotting"])
    footswing_height_cmd = 0.1
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.25

    measured_x_vels = []
    target_x_vels = []
    joint_positions = []

    obs = env.reset()

    i=0

    while(1):
        with torch.no_grad():
            actions = policy(obs)
        env.commands[:, 0] = joy_cmd[0]
        env.commands[:, 1] = joy_cmd[1]
        env.commands[:, 2] = joy_cmd[2]
        env.commands[:, 3] = body_height_cmd
        env.commands[:, 4] = step_frequency_cmd
        env.commands[:, 5:7] = gait
        env.commands[:, 8] = 0.5
        env.commands[:, 9] = footswing_height_cmd
        env.commands[:, 10] = pitch_cmd
        env.commands[:, 11] = roll_cmd
        # env.commands[:, 12] = stance_width_cmd
        obs, rew, done, info = env.step(actions)

        measured_x_vels.append(env.base_lin_vel[0, 0].cpu().numpy())
        joint_positions.append(env.dof_pos[0, :].cpu().numpy())
        target_x_vels.append (joy_cmd[0])

        i=i+1

        # print(env.base_pos[:, 2])

        if(stop):
            break

    # plot target and measured forward velocity
    measured_x_vels = np.array(measured_x_vels)
    target_x_vels = np.array(target_x_vels)
    joint_positions = np.array(joint_positions)
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(12, 5))
    axs[0].plot(np.linspace(0, i * env.dt, i), measured_x_vels, color='black', linestyle="-", label="Measured")
    axs[0].plot(np.linspace(0, i * env.dt, i), target_x_vels, color='black', linestyle="--", label="Desired")
    axs[0].legend()
    axs[0].set_title("Forward Linear Velocity")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Velocity (m/s)")

    # axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), joint_positions, linestyle="-", label="Measured")
    # axs[1].set_title("Joint Positions")
    # axs[1].set_xlabel("Time (s)")
    # axs[1].set_ylabel("Joint Position (rad)")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_go1(headless=False)
