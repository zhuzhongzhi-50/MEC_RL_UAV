import numpy as np
import random
import tensorflow as tf
import datetime
import json
from matplotlib import pyplot as plt

from env import define
from env import environment
import mec_rl_with_uav

print("TensorFlow version: ", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
plt.rcParams['figure.figsize'] = (9, 9)

# 仿真环境大小、各设备数量
map_size = 200
uav_num = 4
server_num = 4
sensor_num = 30

# 各设备的观察、收集半径、移动速度
uav_obs_r = 60

uav_collect_r = 40
server_collect_r = 30

# uav_move_r = 6
#【2023年10月12日】 原代码是将无人机的移动设置为6，这里忽视无人机能够移动的距离，直接设置为最大60
uav_move_r = 6
sensor_move_r = 3

# 训练参数：最大训练轮数、最大的探索步数、学习率、奖励折扣、目标网络的软更新、批次、探索因子
# alpha、beta 目前还没有用到，以后有用可以使用
MAX_EPOCH = 10000
MAX_EP_STEPS = 200
LR_A = 0.001
LR_C = 0.002
GAMMA = 0.85
TAU = 0.8
BATCH_SIZE = 128
alpha = 0.9
beta = 0.1
Epsilon = 0.2
render_freq = 32

# 联合更新参数：联合更新频率、是否进行联合更新、联合更新权重（取1表示不进行联合更新）
up_freq = 8
FL = True
FL_omega = 0

# 设置随机种子
map_seed = 1
rand_seed = 17
np.random.seed(map_seed)
random.seed(map_seed)
tf.random.set_seed(rand_seed)

params = {
    'map_size': map_size,
    'uav_num': uav_num,
    'server_num': server_num,
    'sensor_num': sensor_num,
    'uav_obs_r': uav_obs_r,
    'uav_collect_r': uav_collect_r,
    'server_collect_r': server_collect_r,
    'uav_move_r': uav_move_r,
    'sensor_move_r': sensor_move_r,

    'MAX_EPOCH': MAX_EPOCH,
    'MAX_EP_STEPS': MAX_EP_STEPS,
    'LR_A': LR_A,
    'LR_C': LR_C,
    'GAMMA': GAMMA,
    'TAU': TAU,
    'BATCH_SIZE': BATCH_SIZE,
    'alpha': alpha,
    'beta': beta,
    'Epsilon': Epsilon,
    'learning_seed': rand_seed,
    'env_seed': map_seed,
    'up_freq': up_freq,
    'render_freq': render_freq,
    'FL': FL,
    'FL_omega': FL_omega
}

# 初始化应用环境，将设备数量、设备范围信息，交给define，进行设备属性的初始化定义
mec_world = define.MEC_world(map_size, uav_num, server_num, sensor_num, uav_obs_r, uav_collect_r, server_collect_r, uav_move_r, sensor_move_r)

# 执行神经网络的输出结果
env = environment.MEC_RL_ENV(mec_world)

# 初始化智能体
MAAC = mec_rl_with_uav.MEC_RL_With_Uav(env, TAU, GAMMA, LR_A, LR_C, LR_A, LR_C, BATCH_SIZE, Epsilon)

# 将超参数以JSON格式保存到文件中，方便后续回溯和复现
m_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
f = open('logs/hyperparam/%s.json' % m_time, 'w')
json.dump(params, f)
f.close()

# 开始训练智能体
MAAC.train(MAX_EPOCH, MAX_EP_STEPS, up_freq=up_freq, render=True, render_freq=render_freq, FL=FL, FL_omega=FL_omega)