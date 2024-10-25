

import torch

from go1_gym_learn.utils import split_and_pad_trajectories
#存储器，存储多个Transition对象
class RolloutStorage:
    #存储一个时间步的所有必要信息
    class Transition:
        def __init__(self):
            self.observations = None
            self.privileged_observations = None
            self.observation_histories = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None    # 动作的对数概率
            self.action_mean = None     #策略网络的动作输出均值
            self.action_sigma = None    #策略网络的动作方差输出
            self.env_bins = None    #存储环境的辅助信息

        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, obs_history_shape, actions_shape, device='cpu'):

        self.device = device

        self.obs_shape = obs_shape  #70
        self.privileged_obs_shape = privileged_obs_shape    #2
        self.obs_history_shape = obs_history_shape  #2100
        self.actions_shape = actions_shape  #12

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)  #24，2048，70
        self.privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)    #24，2048，2
        self.observation_histories = torch.zeros(num_transitions_per_env, num_envs, *obs_history_shape, device=self.device)     #24，2048，2100
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)   #选择的当前动作对数概率
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        # 在连续动作空间中，策略网络通常假设动作的概率分布是高斯分布，而 mu 就是这个高斯分布的均值。
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.env_bins = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env  #24 每个环境存储的最大时间步数
        self.num_envs = num_envs    #环境的数量 2048

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)     #copy_:深拷贝
        self.privileged_observations[self.step].copy_(transition.privileged_observations)
        self.observation_histories[self.step].copy_(transition.observation_histories)
        self.actions[self.step].copy_(transition.actions)

        #view:用于重新调整张量的形状。-1 表示这个维度的大小由 PyTorch 自动推断，1 表示目标张量的第二个维度为1
        #将一维张量（2048）调整为二维
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self.env_bins[self.step].copy_(transition.env_bins.view(-1, 1))     #环境辅助信息
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        #从后往前计算
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:    #最后一个时间步，next_values 设置为传入的 last_values
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()   #终止状态则该值为0

            # TD误差（时间差分误差），delta = r + γV(s') - V(s)，它衡量了当前时间步的值函数估计与实际的奖励加上未来状态的值函数估计之间的差异。
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]

            # 使用 GAE 计算advantage，lam为GAE 的衰减因子 ，advantage = δ + γλA(s')     （引入lam用来平滑和降低方差）
            #优势函数A(S,a)=Q(S,a)-V(S).  delta只是一个时间步上的优势估计，只考虑了一个时间步的奖励和下一状态的价值
            #优势函数通常涉及多个时间步的奖励信息，使用 GAE 进行累积和折扣，以减少方差并提高估计的稳定性
            advantage = delta + next_is_not_terminal * gamma * lam * advantage

            #advantage 是基于GAE计算的优势，表示当前动作相对于平均策略的“超额回报”。V(s) 是当前状态的值函数估计
            #return即为长期回报，它融合了即时奖励和未来的累积回报，强化学习的目标需要最大化长期回报
            #优势函数在目标函数的表达式中
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values    #回报减去估计值
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)   #归一化，均值设置为，标准差设置为1

    #计算平均轨迹长度以及奖励平均值
    def get_statistics(self):
        done = self.dones
        done[-1] = 1    #最后一个时间步的所有环境标记为 done
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)   #改变 done 张量的维度顺序，从 (time, envs, 1) 转换为 (envs, time, 1)。
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()
    

    #小批量数据生成器
    #num_mini_batches：小批量数据的数量，控制每个训练周期中分成多少个小批次
    #num_epochs： 每个训练周期内循环的次数
    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env   #2048*24
        mini_batch_size = batch_size // num_mini_batches    #每个小批次的数据大小
        #生成一个长度为 n 的一维张量（随机排列的），n为总的样本数量；不需要计算梯度
        indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)

        #将第0维和第一维合并为一个维度，合并之后observations为24*2048，70
        observations = self.observations.flatten(0, 1)
        privileged_obs = self.privileged_observations.flatten(0, 1)
        obs_history = self.observation_histories.flatten(0, 1)
        critic_observations = observations
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)
        old_env_bins = self.env_bins.flatten(0, 1)

        for epoch in range(num_epochs): #训练的次数
            for i in range(num_mini_batches):   #多少个小批次

                start = i*mini_batch_size   #每个小批次的开始
                end = (i+1)*mini_batch_size #每个小批次的结束
                batch_idx = indices[start:end]  #从生成的随机索引中提取当前小批次的索引范围

                #observations为2048*24  对所有的分为num_mini_batches（4）个小批次，obs_batch为2048*24/4=12288
                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                privileged_obs_batch = privileged_obs[batch_idx]
                obs_history_batch = obs_history[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                env_bins_batch = old_env_bins[batch_idx]
                #yield用于生成一个生成器。与return不同，yield不会终止函数执行，而是保留函数状态，等待下一次调用，直至执行完或return
                yield obs_batch, critic_observations_batch, privileged_obs_batch, obs_history_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
                       old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, None, env_bins_batch

    # for RNNs only
    def reccurent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        #详细见split_and_pad_trajectories说明
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        padded_privileged_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.privileged_observations, self.dones)
        padded_obs_history_trajectories, trajectory_masks = split_and_pad_trajectories(self.observation_histories, self.dones)
        padded_critic_obs_trajectories = padded_obs_trajectories

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i*mini_batch_size
                stop = (i+1)*mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size
                
                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]
                privileged_obs_batch = padded_privileged_obs_trajectories[:, first_traj:last_traj]
                obs_history_batch = padded_obs_history_trajectories[:, first_traj:last_traj]

                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                yield obs_batch, critic_obs_batch, privileged_obs_batch, obs_history_batch, actions_batch, values_batch, advantages_batch, returns_batch, \
                       old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, masks_batch
                
                first_traj = last_traj