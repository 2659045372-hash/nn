
import os
import numpy as np
import tensorflow as tf
import random
from collections import deque

tf.compat.v1.disable_eager_execution()

class RL_QG_agent:
    """
    黑白棋（Reversi）强化学习智能体。
    
    该智能体基于 Q-Learning 算法，并使用卷积神经网络（CNN）作为函数逼近器来估计 Q 值。
    支持经验回放（Experience Replay）和 Epsilon-Greedy 探索策略。
    """
    
    def __init__(self, learning_rate: float = 0.001, reward_decay: float = 0.9, e_greedy: float = 0.9, replace_target_iter: int = 200, memory_size: int = 2000, batch_size: int = 32):
        """
        初始化智能体。

        Args:
            learning_rate: 学习率。
            reward_decay: 奖励折扣因子 (gamma)。
            e_greedy: 最大探索率。
            replace_target_iter: 更新目标网络的步数（当前未显式使用双网络结构，预留）。
            memory_size: 经验回放池的大小。
            batch_size: 训练时的批次大小。
        """
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Reversi")
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 强化学习参数
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.epsilon = 0.1  # 初始探索率，随训练增加
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.learn_step_counter = 0
        
        # 经验回放池
        self.memory = deque(maxlen=self.memory_size)
        
        # TensorFlow组件
        self.sess = None
        self.saver = None
        self.input_states = None
        self.Q_values = None
        self.target_Q = None
        self.loss = None
        self.train_op = None

    def _build_net(self, input_tensor, name):
        """构建神经网络结构"""
        with tf.compat.v1.variable_scope(name):
            # 卷积层1
            conv1 = tf.compat.v1.layers.conv2d(
                inputs=input_tensor,
                filters=32,
                kernel_size=3,
                padding="same",
                activation=tf.nn.relu,
                name="conv1"
            )
            # 卷积层2
            conv2 = tf.compat.v1.layers.conv2d(
                inputs=conv1,
                filters=64,
                kernel_size=3,
                padding="same",
                activation=tf.nn.relu,
                name="conv2"
            )
            # 扁平化
            flat = tf.compat.v1.layers.flatten(conv2)
            # 全连接层
            dense = tf.compat.v1.layers.dense(
                inputs=flat,
                units=512,
                activation=tf.nn.relu,
                name="dense1"
            )
            # 输出层
            q_values = tf.compat.v1.layers.dense(
                inputs=dense,
                units=64,
                name="q_values"
            )
        return q_values

    def init_model(self):
        """构建卷积神经网络模型"""
        self.sess = tf.compat.v1.Session()
        
        # 输入：[批次大小, 8, 8, 3]
        self.input_states = tf.compat.v1.placeholder(
            tf.float32, shape=[None, 8, 8, 3], name="input_states"
        )
        self.target_Q = tf.compat.v1.placeholder(
            tf.float32, shape=[None, 64], name="target_q"
        )
        
        # 构建评估网络 (eval_net)
        self.Q_values = self._build_net(self.input_states, "eval_net")
        
        # 损失函数：均方误差
        self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q_values))
        
        # 优化器
        self.train_op = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.loss)
        
        # 初始化变量
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver()

    def store_transition(self, s, a, r, s_, done):
        """存储经验"""
        self.memory.append((s, a, r, s_, done))

    def learn(self):
        """从经验回放池中学习"""
        if len(self.memory) < self.batch_size:
            return

        # 随机采样一个批次
        batch = random.sample(self.memory, self.batch_size)
        batch_s = np.array([x[0] for x in batch])
        batch_a = np.array([x[1] for x in batch])
        batch_r = np.array([x[2] for x in batch])
        batch_s_ = np.array([x[3] for x in batch])
        batch_done = np.array([x[4] for x in batch])

        # 计算当前 Q 值
        q_eval = self.sess.run(self.Q_values, feed_dict={self.input_states: batch_s})
        # 计算下一状态的 Q 值
        q_next = self.sess.run(self.Q_values, feed_dict={self.input_states: batch_s_})

        # 更新目标 Q 值
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        
        # Q-Learning 更新公式: Q(s,a) = r + gamma * max(Q(s',a'))
        # 如果游戏结束，则 Q(s,a) = r
        max_q_next = np.max(q_next, axis=1)
        q_target[batch_index, batch_a] = batch_r + self.gamma * max_q_next * (1 - batch_done)

        # 执行优化
        _, cost = self.sess.run(
            [self.train_op, self.loss],
            feed_dict={
                self.input_states: batch_s,
                self.target_Q: q_target
            }
        )

        # 逐渐增加 epsilon
        if self.epsilon < self.epsilon_max:
            self.epsilon += 0.001
        
        self.learn_step_counter += 1

    def place(self, state, enables):
        """根据当前状态和合法动作选择最优落子位置 (Epsilon-Greedy)"""
        # 状态预处理
        state_input = np.array(state).reshape(1, 8, 8, 3).astype(np.float32)
        
        # Epsilon-Greedy 探索
        if random.random() < self.epsilon:
            # 贪婪选择：计算所有位置的 Q 值
            q_vals = self.sess.run(self.Q_values, feed_dict={self.input_states: state_input})
            legal_q = q_vals[0][enables]
            
            # 选择 Q 值最大的合法动作
            max_q = np.max(legal_q)
            best_indices = np.where(legal_q == max_q)[0]
            action = enables[np.random.choice(best_indices)]
        else:
            # 探索：随机选择一个合法动作
            action = np.random.choice(enables)
            
        return action

    def save_model(self):
        """保存模型参数"""
        try:
            model_path = os.path.join(self.model_dir, 'parameter.ckpt')
            self.saver.save(self.sess, model_path)
            print("模型已保存至", self.model_dir)
        except Exception as e:
            print("保存模型时出错:", e)

    def load_model(self):
        """加载模型参数"""
        if self.sess is None:
            self.init_model()  # 未初始化则先构建模型
        
        model_path = os.path.join(self.model_dir, 'parameter.ckpt')
        if not os.path.exists(model_path + '.index'):
            print("模型文件不存在，使用初始化模型")
            return
        
        try:
            self.saver.restore(self.sess, model_path)
            print("模型已从", self.model_dir, "加载")
        except Exception as e:
            print("加载模型时出错:", e)