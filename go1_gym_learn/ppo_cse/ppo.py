# from collections.abc import generator

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from params_proto import PrefixProto

from go1_gym_learn.ppo_cse import ActorCritic
from go1_gym_learn.ppo_cse import RolloutStorage
from go1_gym_learn.ppo_cse import caches


class PPO_Args(PrefixProto):
    # algorithm
    value_loss_coef = 1.0   #价值函数损失的系数      PPO算法的目标函数包括三个部分，策略函数的损失（新旧策略概率比值乘以优势函数，鼓励表现较好的动作）
    # 价值函数的损失（预测价值与实际价值之间的差异，平方损失），以及一个策略模型的熵（鼓励探索）
    use_clipped_value_loss = True   #是否使用截断的价值损失
    clip_param = 0.2    #控制截断的幅度
    entropy_coef = 0.01      # 熵系数，用于鼓励策略探索更多的动作。较高的熵值意味着策略会更倾向于探索，而不是只选择最有把握的动作。
    num_learning_epochs = 5     #每个 PPO 更新的学习轮数。在每次更新中，策略会用收集到的数据进行多次更新，这个参数决定了更新的次数。
    num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches    mini-batch 的数量。数据会被划分为多个 mini-batch，用于梯度下降的每一步
    learning_rate = 1.e-3  # 5.e-4      #学习率
    adaptation_module_learning_rate = 1.e-3     #适应模块的学习率
    num_adaptation_module_substeps = 1      #适应模块的子步数，指在适应过程中每步内进行多少次更新
    schedule = 'adaptive'  # could be adaptive, fixed       #学习率调整策略，adaptive或kp、kd
    gamma = 0.99    # 折扣因子，决定了算法在更新策略时，如何看待未来的奖励。接近 1 的值表示算法对长期奖励更为重视。
    lam = 0.95      #用于优势函数估计的平滑参数，称为 GAE（广义优势估计）参数，用来平衡偏差和方差
    desired_kl = 0.01       #目标 Kullback-Leibler (KL) 散度，用于监控和限制新旧策略之间的差异。如果策略变化过大，可能会调整学习率。
    max_grad_norm = 1.      #梯度裁剪的最大范数，用于防止梯度爆炸，从而稳定训练过程。

    selective_adaptation_module_loss = False        #是否选择性地应用适应模块的损失


class PPO:
    actor_critic: ActorCritic

    def __init__(self, actor_critic, device='cpu'):

        self.device = device

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(device)
        self.storage = None  # initialized later    存储器
        #使用 Adam 优化器对 actor_critic 的参数进行优化，学习率由 PPO_Args.learning_rate 指定。
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=PPO_Args.learning_rate)
        self.adaptation_module_optimizer = optim.Adam(self.actor_critic.parameters(),
                                                      lr=PPO_Args.adaptation_module_learning_rate)
        if self.actor_critic.decoder:
            self.decoder_optimizer = optim.Adam(self.actor_critic.parameters(),
                                                          lr=PPO_Args.adaptation_module_learning_rate)
        self.transition = RolloutStorage.Transition()   #初始化一个 transition 对象，存储单步交互的状态、动作、奖励等数据。

        self.learning_rate = PPO_Args.learning_rate

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape, obs_history_shape,
                     action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape,
                                      obs_history_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, privileged_obs, obs_history):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs_history).detach()
        self.transition.values = self.actor_critic.evaluate(obs_history, privileged_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = obs
        self.transition.privileged_observations = privileged_obs
        self.transition.observation_histories = obs_history
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        #将infos字典中的env_bins信息保存到self.transition.env_bins中
        self.transition.env_bins = infos["env_bins"]

        # Bootstrapping on time outs
        #torch.squeeze 移除张量维度为1的多余维度
        #unsqueeze(1) ：将一维张量扩展为二维张量、
        #超时，系统没有给出明确的终止奖励，需要使用状态价值弥补超时奖励的不足
        if 'time_outs' in infos:
            self.transition.rewards += PPO_Args.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    #todo
    def compute_returns(self, last_critic_obs, last_critic_privileged_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs, last_critic_privileged_obs).detach()
        self.storage.compute_returns(last_values, PPO_Args.gamma, PPO_Args.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_adaptation_module_loss = 0
        mean_decoder_loss = 0
        mean_decoder_loss_student = 0
        mean_adaptation_module_test_loss = 0
        mean_decoder_test_loss = 0
        mean_decoder_test_loss_student = 0
        # generator = self.storage.reccurent_mini_batch_generator(PPO_Args.num_mini_batches, PPO_Args.num_learning_epochs)
        # for obs_batch, critic_obs_batch, privileged_obs_batch, obs_history_batch, actions_batch, values_batch, advantages_batch, returns_batch, \
        #                old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, masks_batch in generator:
        generator = self.storage.mini_batch_generator(PPO_Args.num_mini_batches, PPO_Args.num_learning_epochs)
        for obs_batch, critic_obs_batch, privileged_obs_batch, obs_history_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, masks_batch, env_bins_batch in generator:

            self.actor_critic.act(obs_history_batch, masks=masks_batch)
            #策略更新时更新epoch次，每次有num_mini_batches个批次。
            #第一次进行更新的时候，由于此时策略网络参数未发生改变，因此此处得出的action概率与generator中一致，但是此次进行更新后，参数发生变化，概率便不一致了，
            #即新旧策略概率之比
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(obs_history_batch, privileged_obs_batch, masks=masks_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL    衡量当前策略与旧策略的差异。如果KL散度过大或过小，会调整学习率，以便在训练中保持策略变化的稳定性。
            if PPO_Args.desired_kl != None and PPO_Args.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (
                                torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (
                                2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > PPO_Args.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < PPO_Args.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Surrogate loss  PPO目标函数中的代理损失
            #old action来自于上一个动作，action为本次obs history输入到策略网络中获取的输出
            #原公式是在未裁减代理损失（surrogate）与裁减之后的代理损失（surrogate_clipped）之间取最小值。
            #代码对二者都进行了取反，所以取最大值
            #torch.squeeze移除尺寸为1的维度
            #优势函数见rollout_storage.py，采取一个动作后的回报减去估计值
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - PPO_Args.clip_param,
                                                                               1.0 + PPO_Args.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if PPO_Args.use_clipped_value_loss:
                value_clipped = target_values_batch + \
                                (value_batch - target_values_batch).clamp(-PPO_Args.clip_param,
                                                                          PPO_Args.clip_param)
                #未剪切的值函数损失，平方损失，预测值与真实回报之间的平方差
                value_losses = (value_batch - returns_batch).pow(2)
                #剪切后的值函数损失
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                #todo 为什么max
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            #总的目标函数=代理损失+价值损失+熵损失
            #todo 第二项为什么是加上？ppo公式为减去（value_loss_coef=1）
            loss = surrogate_loss + PPO_Args.value_loss_coef * value_loss - PPO_Args.entropy_coef * entropy_batch.mean()

            # Gradient step
            #梯度是累积的，所以每次进行反向传播前，需要将之前计算的梯度清零
            self.optimizer.zero_grad()
            #反向传播计算梯度
            loss.backward()
            #梯度裁剪。      max_grad_norm：梯度的最大允许范数
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), PPO_Args.max_grad_norm)
            #更新模型参数
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

            data_size = privileged_obs_batch.shape[0]   #12288（2048*24/4）
            num_train = int(data_size // 5 * 4)

            # Adaptation module gradient step
            #更新适应性模块
            for epoch in range(PPO_Args.num_adaptation_module_substeps):

                adaptation_pred = self.actor_critic.adaptation_module(obs_history_batch)    #适应性模块预测值
                with torch.no_grad():
                    adaptation_target = privileged_obs_batch    #目标值
                    # residual = (adaptation_target - adaptation_pred).norm(dim=1)
                    # caches.slot_cache.log(env_bins_batch[:, 0].cpu().numpy().astype(np.uint8),
                    #                       sysid_residual=residual.cpu().numpy())

                selection_indices = torch.linspace(0, adaptation_pred.shape[1]-1, steps=adaptation_pred.shape[1], dtype=torch.long)
                if PPO_Args.selective_adaptation_module_loss:
                    # mask out indices corresponding to swing feet
                    selection_indices = 0

                #MSE均方误差作为损失函数来计算预测值和目标值之间的误差
                adaptation_loss = F.mse_loss(adaptation_pred[:num_train, selection_indices], adaptation_target[:num_train, selection_indices])
                #测试集，占20%
                adaptation_test_loss = F.mse_loss(adaptation_pred[num_train:, selection_indices], adaptation_target[num_train:, selection_indices])



                self.adaptation_module_optimizer.zero_grad()
                adaptation_loss.backward()
                self.adaptation_module_optimizer.step()

                mean_adaptation_module_loss += adaptation_loss.item()
                mean_adaptation_module_test_loss += adaptation_test_loss.item()

        num_updates = PPO_Args.num_learning_epochs * PPO_Args.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_adaptation_module_loss /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_decoder_loss /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_decoder_loss_student /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_adaptation_module_test_loss /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_decoder_test_loss /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_decoder_test_loss_student /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_decoder_loss, mean_decoder_loss_student, mean_adaptation_module_test_loss, mean_decoder_test_loss, mean_decoder_test_loss_student
