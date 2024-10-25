import torch
import numpy as np
from go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *
from isaacgym import gymapi

class CoRLRewards:
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.env.commands[:, :2] - self.env.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.env.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.env.cfg.rewards.tracking_sigma_yaw)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.env.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.env.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.env.torques), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.env.last_dof_vel - self.env.dof_vel) / self.env.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.env.last_actions - self.env.actions), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.env.contact_forces[:, self.env.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.env.dof_pos - self.env.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.env.dof_pos - self.env.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_jump(self):
        # reference_heights = 0
        # body_height = self.env.base_pos[:, 2] - reference_heights
        body_height = self.env.base_pos[:, 2]
        jump_height_target = self.env.commands[:, 3] + self.env.cfg.rewards.base_height_target
        reward = - torch.square(body_height - jump_height_target)
        return reward

    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1)
        desired_contact = self.env.desired_contact_states

        reward = 0
        for i in range(4):
            reward += - (1 - desired_contact[:, i]) * (
                        1 - torch.exp(-1 * foot_forces[:, i] ** 2 / self.env.cfg.rewards.gait_force_sigma))
        return reward / 4

    def _reward_tracking_contacts_shaped_vel(self):
        foot_velocities = torch.norm(self.env.foot_velocities, dim=2).view(self.env.num_envs, -1)
        desired_contact = self.env.desired_contact_states
        reward = 0
        for i in range(4):
            reward += - (desired_contact[:, i] * (
                        1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / self.env.cfg.rewards.gait_vel_sigma)))
        return reward / 4

    def _reward_dof_pos(self):
        # Penalize dof positions
        return torch.sum(torch.square(self.env.dof_pos - self.env.default_dof_pos), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.env.dof_vel), dim=1)

    def _reward_action_smoothness_1(self):
        # Penalize changes in actions
        diff = torch.square(self.env.joint_pos_target[:, :self.env.num_actuated_dof] - self.env.last_joint_pos_target[:, :self.env.num_actuated_dof])
        diff = diff * (self.env.last_actions[:, :self.env.num_dof] != 0)  # ignore first step
        return torch.sum(diff, dim=1)

    def _reward_action_smoothness_2(self):
        # Penalize changes in actions
        diff = torch.square(self.env.joint_pos_target[:, :self.env.num_actuated_dof] - 2 * self.env.last_joint_pos_target[:, :self.env.num_actuated_dof] + self.env.last_last_joint_pos_target[:, :self.env.num_actuated_dof])
        diff = diff * (self.env.last_actions[:, :self.env.num_dof] != 0)  # ignore first step
        diff = diff * (self.env.last_last_actions[:, :self.env.num_dof] != 0)  # ignore second step
        return torch.sum(diff, dim=1)

    def _reward_feet_slip(self):
        contact = self.env.contact_forces[:, self.env.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.env.last_contacts)
        self.env.last_contacts = contact
        foot_velocities = torch.square(torch.norm(self.env.foot_velocities[:, :, 0:2], dim=2).view(self.env.num_envs, -1))
        rew_slip = torch.sum(contact_filt * foot_velocities, dim=1)
        return rew_slip

    def _reward_feet_contact_vel(self):
        reference_heights = 0
        near_ground = self.env.foot_positions[:, :, 2] - reference_heights < 0.03
        foot_velocities = torch.square(torch.norm(self.env.foot_velocities[:, :, 0:3], dim=2).view(self.env.num_envs, -1))
        rew_contact_vel = torch.sum(near_ground * foot_velocities, dim=1)
        return rew_contact_vel

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        #对机器人脚部接触力过大的情况进行惩罚
        return torch.sum((torch.norm(self.env.contact_forces[:, self.env.feet_indices, :],
                                     dim=-1) - self.env.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_feet_clearance_cmd_linear(self):
        #计算脚部在特定步态下的高度偏差，旨在让机器人达到预期的脚步抬高高度
        phases = 1 - torch.abs(1.0 - torch.clip((self.env.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = (self.env.foot_positions[:, :, 2]).view(self.env.num_envs, -1)# - reference_heights
        target_height = self.env.commands[:, 9].unsqueeze(1) * phases + 0.02 # offset for foot radius 2cm
        rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.env.desired_contact_states)
        return torch.sum(rew_foot_clearance, dim=1)

    def _reward_feet_impact_vel(self):
        #惩罚脚部与地面接触时的过高速度
        prev_foot_velocities = self.env.prev_foot_velocities[:, :, 2].view(self.env.num_envs, -1)
        contact_states = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1) > 1.0

        rew_foot_impact_vel = contact_states * torch.square(torch.clip(prev_foot_velocities, -100, 0))

        return torch.sum(rew_foot_impact_vel, dim=1)


    def _reward_collision(self):
        # Penalize collisions on selected bodies
        #该函数对机器人特定部位的碰撞进行惩罚
        return torch.sum(1. * (torch.norm(self.env.contact_forces[:, self.env.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)

    def _reward_orientation_control(self):
        # Penalize non flat base orientation
        #对机器人的姿态控制进行奖励      通过四元数运算计算期望的重力方向，并与实际的重力投影差异进行比较，从而对不符合期望姿态的情况进行惩罚。
        roll_pitch_commands = self.env.commands[:, 10:12]
        quat_roll = quat_from_angle_axis(-roll_pitch_commands[:, 1],
                                         torch.tensor([1, 0, 0], device=self.env.device, dtype=torch.float))
        quat_pitch = quat_from_angle_axis(-roll_pitch_commands[:, 0],
                                          torch.tensor([0, 1, 0], device=self.env.device, dtype=torch.float))

        desired_base_quat = quat_mul(quat_roll, quat_pitch)
        desired_projected_gravity = quat_rotate_inverse(desired_base_quat, self.env.gravity_vec)

        return torch.sum(torch.square(self.env.projected_gravity[:, :2] - desired_projected_gravity[:, :2]), dim=1)

    def _reward_raibert_heuristic(self):
        #基于Raibert启发式算法对步态进行奖励，核心思想是将机器人分解为几个简单的子任务，通过调节身体和腿的运动来实现平衡和移动
        #控制器分为三个部分
        #平衡控制（调整支撑腿的位置，使得机器人的身体能够保持在目标位置上），通常通过控制质心的水平位置来实现。
        #速度控制：调节腿部的摆动来控制机器人的前向速度。根据速度需求调整支撑腿在地面上的着地点
        #腿的摆动控制：决定腿在摆动阶段的轨迹，以确保下一步落脚点能实现目标位置

        #落地点调整：下一步落脚点的位置是根据当前速度和目标速度来调整的
        #当机器人速度过快时，算法会选择一个更远的落脚点，以便在下一步时减少速度。   todo why
        #当机器人速度过慢时，落脚点会相对靠近，以加速机器人。

        #cur_footsteps_translated：当前机器人各脚的位置与机器人底座（base）的相对位置。
        #self.env.foot_positions是机器人的脚位置，self.env.base_pos是底座位置，通过相减获得脚相对底座的位置。
        cur_footsteps_translated = self.env.foot_positions - self.env.base_pos.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.env.num_envs, 4, 3, device=self.env.device)
        for i in range(4):
            #四只脚相对于机器人的身体框架（body frame）的位置
            footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.env.base_quat),
                                                              cur_footsteps_translated[:, i, :])

        # nominal positions: [FR, FL, RR, RL]
        #desired_stance_width：期望的步伐宽度
        #desired_ys_nom：每只脚的期望横向（Y轴）位置（中心为原点）
        if self.env.cfg.commands.num_commands >= 13:
            desired_stance_width = self.env.commands[:, 12:13]
            desired_ys_nom = torch.cat([desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], dim=1)
        else:
            desired_stance_width = 0.3
            desired_ys_nom = torch.tensor([desired_stance_width / 2,  -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], device=self.env.device).unsqueeze(0)

        #desired_stance_length：期望的步伐长度
        #desired_xs_nom：每只脚的期望纵向（X轴）位置
        if self.env.cfg.commands.num_commands >= 14:
            desired_stance_length = self.env.commands[:, 13:14]
            desired_xs_nom = torch.cat([desired_stance_length / 2, desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], dim=1)
        else:
            #luyx changed
            desired_stance_length = 0.38
            desired_xs_nom = torch.tensor([desired_stance_length / 2,  desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], device=self.env.device).unsqueeze(0)

        # raibert offsets   Raibert偏移量计算
        #phase：与步伐相位相关的值
        #foot_indices：0～1，0-刚刚触地，0.5：半空中
        phases = torch.abs(1.0 - (self.env.foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = self.env.commands[:, 4]   #步伐频率

        #仅计算x方向线速度和yaw，本质上并为考虑到y方向线速度，其通过yaw间接表示


        yaw_vel_des = self.env.commands[:, 2:3]
        #原代码：
        # x_vel_des = self.env.commands[:, 0:1]
        # y_vel_des = yaw_vel_des * desired_stance_length / 2
        #修改后：
        x_vel_des = self.env.commands[:, 0:1] + yaw_vel_des * desired_stance_width / 2
        y_vel_des = self.env.commands[:, 1:2] + yaw_vel_des * desired_stance_length / 2


        #y_vel_des计算的是由旋转带来的y方向速度（详细可自己推导），与command中的y vel无关
        #仅使用该y vel，不考虑command中的速度，最后也可以训练出侧移
        # y_vel_des = yaw_vel_des * desired_stance_length / 2     #todo 为什么不直接用command中的y vel

        #可简单理解为，步频越快，则偏移量越小。速度越大，偏移量越大。再乘以一个参数。
        # #phases：越大，则当前腿摆动剩余时间越大（phases计算时使用了-），则偏移量越大
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1     #转弯时，前面的脚与后面的脚移动方向应该相反
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))


        #在原始期望步伐位置的基础上加上Raibert偏移量，得到新的期望步伐位置
        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        #最终期望步伐位置的组合，包含X和Y两个方向
        #unsqueeze：在第二维增加一个维度，变为num_envs x 4 x 1（4只脚）
        #相加过后变为num_envs x 4 x 2
        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        return reward