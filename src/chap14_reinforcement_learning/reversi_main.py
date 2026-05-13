
# ==============================================
# 黑白棋（Reversi/Othello）强化学习训练代码
# 核心逻辑：黑棋采用随机策略，白棋采用自定义强化学习智能体（RL_QG_agent）
# 训练流程：环境注册→环境创建→智能体初始化→多局对战训练→结果统计→模型保存
# ==============================================

# 导入基础工具库
import random  # 用于黑棋的随机落子策略
import gym     # OpenAI Gym：强化学习环境标准框架，提供环境创建、交互接口
from gym.envs.registration import register  # Gym环境注册函数，用于自定义环境注册
import numpy as np  # 数值计算库，用于棋盘状态统计（如得分计算）

# 导入自定义模块
from gym.envs.reversi.reversi import ReversiEnv  # 自定义黑白棋环境类（实现棋盘规则、状态管理等）
from RL_QG_agent import RL_QG_agent  # 自定义强化学习智能体类（实现Q学习/其他RL算法）

# ==============================================
# 第一步：注册自定义黑白棋环境
# Gym要求自定义环境必须先注册，才能通过gym.make()创建实例
# ==============================================
register(
    id='Reversi8x8-v0',  # 环境唯一标识符（后续创建环境时使用）
    entry_point='gym.envs.reversi.reversi:ReversiEnv',  # 环境类的路径（包.模块:类名）
    kwargs={  # 传递给ReversiEnv类的初始化参数
        'player_color': 'black',  # 初始玩家颜色（黑棋先行）
        'opponent': 'random',     # 对手类型（此处为随机策略对手，即黑棋是随机玩家）
        'observation_type': 'numpy3c',  # 观测数据类型：3通道numpy数组（可能分别存储黑棋、白棋、空位置）
        'illegal_place_mode': 'lose',   # 非法落子处理方式：直接判负
        'board_size': 8  # 棋盘尺寸（8x8标准黑白棋）
    },
    max_episode_steps=1000,  # 每局最大步数限制（防止无限循环）
)

# 验证环境是否注册成功
envs = [spec.id for spec in gym.envs.registry.all()]  # 获取所有已注册的环境ID列表
print("Reversi8x8-v0 是否注册成功：", 'Reversi8x8-v0' in envs)  # 打印注册结果

# ==============================================
# 第二步：创建黑白棋环境实例
# 基于已注册的环境ID，创建可交互的环境对象
# ==============================================
env = gym.make(
    'Reversi8x8-v0',  # 目标环境ID（必须与注册时一致）
    player_color='black',  # 覆盖注册时的参数：初始玩家为黑棋
    opponent='random',     # 覆盖注册时的参数：对手为随机策略
    observation_type='numpy3c',  # 观测类型：3通道numpy数组
    illegal_place_mode='lose'    # 非法落子直接判负
)

# ==============================================
# 第三步：初始化强化学习智能体（白棋玩家）
# ==============================================
agent = RL_QG_agent()  # 实例化自定义RL智能体（控制白棋）
agent.init_model()     # 初始化智能体的模型（如Q表、神经网络等）
agent.load_model()     # 加载预训练模型（若存在，可基于历史模型继续训练）

# ==============================================
# 第四步：设置训练参数
# ==============================================
max_epochs = 1000  # 增加训练总局数
render_interval = 100  # 减少渲染频率以加快训练速度

# ==============================================
# 第五步：训练主循环（核心逻辑）
# ==============================================
for i_episode in range(max_epochs):
    observation = env.reset()
    state_white = None
    action_white = None
    
    for t in range(100):
        # --- 黑棋回合（随机策略） ---
        enables = env.possible_actions
        if not enables:
            action_black = env.board_size**2 + 1
        else:
            action_black = random.choice(enables)
        
        # 执行黑棋动作，得到白棋回合开始时的观测
        observation, reward, done, info = env.step(action_black)
        
        # 如果上一个白棋动作还在等待结果，则存储它的经验
        # 注意：这里的 reward 是白棋动作后环境给出的，或者黑棋动作后环境给出的
        # 实际开发中需要根据环境定义的奖励机制来调整
        if state_white is not None:
            # 存储 (s, a, r, s', done)
            # 这里简化处理：白棋动作后的 reward 和 next_state
            agent.store_transition(state_white, action_white, reward, observation, done)
            agent.learn()
        
        if done:
            break

        # --- 白棋回合（强化学习智能体） ---
        if i_episode % render_interval == 0:
            env.render()
        
        enables = env.possible_actions
        if not enables:
            action_white = env.board_size ** 2 + 1
            state_white = None # pass 动作不参与训练或简化处理
        else:
            state_white = observation # 记录白棋看到的状态
            action_white = agent.place(observation, enables)
            
            # 执行白棋动作
            observation, reward, done, info = env.step(action_white)
            
            if done:
                # 游戏结束，存储最后的经验
                agent.store_transition(state_white, action_white, reward, observation, done)
                agent.learn()
                break
    
    # 定期保存模型
    if (i_episode + 1) % 100 == 0:
        agent.save_model()

    # 每 10 局打印一次结果
    if (i_episode + 1) % 10 == 0:
        black_score = np.sum(env.board == 1)
        white_score = np.sum(env.board == -1)
        result = "白棋胜" if white_score > black_score else ("黑棋胜" if black_score > white_score else "平局")
        print(f"Episode {i_episode+1}: 黑 {black_score} vs 白 {white_score} ({result}) | Epsilon: {agent.epsilon:.2f}")

# ==============================================
# 第六步：训练结束后处理
# ==============================================
agent.save_model()  # 保存训练后的智能体模型（覆盖原有模型或保存为新文件）
env.close()         # 关闭环境，释放资源（如渲染窗口、内存等）
print(f"\n训练完成！共进行 {max_epochs} 局对战")
