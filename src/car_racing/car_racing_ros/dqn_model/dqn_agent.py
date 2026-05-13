"""
================================================================================
DQN 智能体实现
================================================================================
继承自 BaseAgent，只包含 DQN 特有的更新逻辑。

DQN (Deep Q-Network) 算法原理:
================================================================================
核心思想: 用深度神经网络来逼近Q值函数

Q-Learning 更新公式:
    Q(s, a) = Q(s, a) + α * (r + γ * max(Q(s', a')) - Q(s, a))
    
其中:
- Q(s, a): 状态s下动作a的Q值
- α: 学习率
- r: 当前奖励
- γ: 折扣因子
- max(Q(s', a')): 下一状态的最大Q值

DQN的改进:
1. 使用深度神经网络逼近Q函数
2. 经验回放: 打破样本时间相关性
3. 目标网络: 提供稳定的学习目标

损失函数:
    Loss = MSE(Q(s, a), r + γ * max(Q_target(s', a')))
================================================================================
"""
import torch
import numpy as np
from base_agent import BaseAgent, BaseDQNNetwork, SkipFrame, plot_rewards


class DQNAgent(BaseAgent):
    """
    标准 DQN (Deep Q-Network) 智能体实现。
    
    该智能体实现了经典 DQN 算法的核心组件，包括：
    - 策略网络 (Policy Network): 用于选择动作并计算当前 Q 值。
    - 目标网络 (Target Network): 周期性同步自策略网络，用于提供稳定的 Q 值目标。
    - 经验回放 (Experience Replay): 通过采样历史经验打破样本间的时间相关性。
    
    特点:
    - 使用固定的目标网络更新（默认每 5000 步同步一次）。
    - 支持 Dueling DQN 架构（通过配置开启）。
    - 支持 Double DQN 逻辑（通过配置开启）。
    
    与 Double DQN 的主要区别:
    - DQN: 使用目标网络同时进行动作选择和价值评估。
    - Double DQN: 使用策略网络选择动作，使用目标网络进行价值评估，从而减少 Q 值的过估计。
    """
    
    def _build_networks(self):
        """
        初始化并构建神经网络。
        
        根据 hyperparameters 中的 'dueling' 配置选择是否使用 Dueling 架构。
        初始化策略网络和目标网络，并将它们移动到指定的计算设备（CPU/GPU）。
        """
        dueling = bool(self.hyperparameters.get('dueling', True))
        self.policy_net = BaseDQNNetwork(self.state_shape, self.action_n, dueling=dueling).float()
        self.frozen_net = BaseDQNNetwork(self.state_shape, self.action_n, dueling=dueling).float()
        self.frozen_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net = self.policy_net.to(self.device)
        self.frozen_net = self.frozen_net.to(self.device)
    
    def update_net(self, batch_size):
        """
        执行 DQN 的一次梯度更新步骤。
        
        参数:
            batch_size (int): 每次更新从经验回放池中采样的样本数量。
            
        返回:
            tuple: (平均 Q 值, 损失值)
            
        算法流程:
        1. 从回放缓冲区采样一批 (s, a, r, s', done) 经验。
        2. 计算当前状态 s 下动作 a 的估计 Q 值 (使用策略网络)。
        3. 根据是否使用 Double DQN 计算目标 Q 值 (使用目标网络评估)。
        4. 计算 MSE 损失并执行反向传播。
        5. 检查是否达到同步目标网络的步数阈值。
        """
        self.n_updates += 1
        states, actions, rewards, new_states, terminateds = self.get_samples(batch_size)
        use_double_q = bool(self.hyperparameters.get("double_q", False))
        max_grad_norm = self.hyperparameters.get("max_grad_norm", None)

        if self.use_amp:
            from torch.cuda.amp import autocast
            with torch.no_grad(), autocast():
                if use_double_q:
                    next_actions = self.policy_net(new_states).argmax(dim=1)
                    next_q = self.frozen_net(new_states).gather(1, next_actions.unsqueeze(1)).view(-1)
                else:
                    next_q = self.frozen_net(new_states).max(1)[0]
                target_q = rewards + (1 - terminateds.float()) * self.gamma * next_q

            self.optimizer.zero_grad(set_to_none=True)
            with autocast():
                current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).view(-1)
                loss = self.loss_fn(current_q, target_q)
            self.scaler.scale(loss).backward()
            if max_grad_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), float(max_grad_norm))
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            q_values = self.policy_net(states)
            current_q = q_values.gather(1, actions.unsqueeze(1)).view(-1)
            with torch.no_grad():
                if use_double_q:
                    next_actions = self.policy_net(new_states).argmax(dim=1)
                    next_q = self.frozen_net(new_states).gather(1, next_actions.unsqueeze(1)).view(-1)
                else:
                    next_q = self.frozen_net(new_states).max(1)[0]
                target_q = rewards + (1 - terminateds.float()) * self.gamma * next_q
            loss = self.loss_fn(current_q, target_q)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), float(max_grad_norm))
            self.optimizer.step()
        
        # -------------------------------------------------------------------------
        # 步骤4: 定期同步目标网络
        # -------------------------------------------------------------------------
        if self.n_updates % self.hyperparameters.get('target_update', 5000) == 0:
            self.sync_target_net()
        
        return current_q.mean().item(), float(loss.item())


# ============================================================================
# 兼容性别名 (为了与旧代码兼容)
# ============================================================================
Agent = DQNAgent
plot_reward = plot_rewards
