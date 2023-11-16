import numpy as np
import gym
from gym import spaces
import numpy as np
from .space_def import circle_space
from .space_def import onehot_space
from .space_def import sum_space
from gym.envs.registration import EnvSpec
import logging
from matplotlib import pyplot as plt
from IPython import display
import random
from env import traffic
from scipy.stats import rankdata
logging.basicConfig(level=logging.WARNING)

# 得到一个表示圆形的点的集合，用于可视化设备的覆盖范围
def get_circle_plot(pos, r):
    x_c = np.arange(-r, r, 0.01)
    up_y = np.sqrt(r**2 - np.square(x_c))
    down_y = - up_y
    x = x_c + pos[0]
    y1 = up_y + pos[1]
    y2 = down_y + pos[1]
    return [x, y1, y2]

class MEC_RL_ENV(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self, world):
        # MEC场景、场景大小、场景中传感器的位置状态
        self.world = world
        self.map_size = self.world.map_size
        self.DS_state = self.world.DS_state

        # 各设备、各设备数量、观察、收集、移动半径
        self.uavs = self.world.uavs
        self.servers = self.world.servers
        self.sensors = self.world.sensors

        self.uav_num = self.world.uav_count
        self.server_num = self.world.server_count
        self.sensor_num = self.world.sensor_count

        self.uav_obs_r = world.uav_obs_r
        self.uav_collect_r = world.uav_collect_r
        self.server_collect_r = world.server_collect_r
        self.uav_move_r = world.uav_move_r
        self.sensor_move_r = world.sensor_move_r

        # 进行初始化渲染，弹出图片
        # self.render()
    
    # 由于流程的设计理念原因（卸载决策需要考虑到无人机的移动）
    # 没有在这里执行 uav_action，而是在 mec_rl_uav.py 中进行了执行
    def step(self, uav_action, sensor_action, if_random):
        obs = []
        uav_reward = []
        uav_rewards = []

        #【 第一步： 执行动作】
        logging.info("set actions")
        # 移动终端的卸载动作和随机移动，写在了 define.py 中
        # 无人机的动作移动执行，写在了 mec_rl_uav.py 中，为了获得无人机的移动后的位置，使用移动后的位置，作为网络的输入得到卸载决策

        # 区分 网络生成的决策、随机生成的决策，将决策放入到 sensor.action.offload 中
        if(not if_random): 
            count_i = 0
            for i, sensor in enumerate(self.sensors):
                # 将之前的数据进行清空
                sensor.action.offload = []
                points = [
                    sensor.position,
                    self.servers[0].position,
                    self.servers[1].position,
                    self.servers[2].position,
                    self.servers[3].position,
                    self.uavs[0].position,
                    self.uavs[1].position,
                    self.uavs[2].position,
                    self.uavs[3].position,
                ]
                distances = []
                # 计算第一个点到其他点之间的距离
                for point in points:
                    distance = np.linalg.norm(np.array(points[0]) - np.array(point))
                    distances.append(distance)
                # 针对没有被任何设备覆盖的传感器，直接跳过
                if (
                    distances[1] > 30 and
                    distances[2] > 30 and
                    distances[3] > 30 and
                    distances[4] > 30 and
                    distances[5] > 40 and
                    distances[6] > 40 and
                    distances[7] > 40 and
                    distances[8] > 40
                ):
                    continue
                sensor.action.offload = sensor_action[count_i]
                count_i += 1
        else: 
            for i, sensor in enumerate(self.sensors):
                sensor.action.offload = sensor_action[i]

        # 调用 define.py 开始卸载决策和随机移动
        self.world.step()

        # 关于这两个值得不一样，我表示非常得差异！
        # print(self.DS_state)
        # print(self.world.DS_state)

        #【 第二步：观察新状态 】
        logging.info("uav observation")
        # 获得新的，new observation，也就是强化学习那里需要使用的 S(t+1)
        #【问题】关于这个 状态有一些问题
        for uav in self.uavs:
            obs.append(self.get_uav_obs(uav))

        processed_age = 10/(self.world.all_sensors_age+0.01)
        age_reward = 0

        # 200 以上的 len
        if processed_age < 0.05:
            age_reward = 0.1
        # 100 - 200
        if processed_age >= 0.05 and processed_age < 0.1:
            age_reward = 0.3
        # 50 - 100
        if processed_age >= 0.1 and processed_age < 0.2:
            age_reward = 0.6
        if processed_age >= 0.2 and processed_age < 0.3:
            age_reward = 0.7
        if processed_age >= 0.3 and processed_age < 0.4:
            age_reward = 0.8
        if processed_age >= 0.4 and processed_age < 0.5:
            age_reward = 0.8
        if processed_age >= 0.5:
            age_reward = 1
        
        #【 第三步：获得奖励 】
        logging.info("get reward")
        # 传感器的奖励在 define.py 文件中
        # 无人机的奖励在 environment.py 本文件中
        # 计算无人机执行移动决策奖励：考虑无人机整体的覆盖率，四个无人机具有相同的奖励
        uav_reward = self.get_uav_reward()
        for uav in self.uavs:
            all_uav_reward = 0.5*uav_reward + 0.5*age_reward
            # all_uav_reward = age_reward
            # all_uav_reward = uav_reward
            uav_rewards.append(round(all_uav_reward, 3))
        
        return obs, uav_rewards, self.world.sensor_delay
    
    # 针对某个特定的无人机，所观察到的信息数据
    def get_uav_obs(self, uav):
        obs = np.zeros([uav.uav_obs_r * 2 + 1, uav.uav_obs_r * 2 + 1, 2])
        # 左上点
        lu = [max(0, uav.position[0] - uav.uav_obs_r),
              min(self.map_size, uav.position[1] + uav.uav_obs_r + 1)]
        # 右下点
        rd = [min(self.map_size, uav.position[0] + uav.uav_obs_r + 1),
              max(0, uav.position[1] - uav.uav_obs_r)]
        # 观察图的位置
        ob_lu = [uav.uav_obs_r - uav.position[0] + lu[0],
                 uav.uav_obs_r - uav.position[1] + lu[1]]
        ob_rd = [uav.uav_obs_r + rd[0] - uav.position[0],
                 uav.uav_obs_r + rd[1] - uav.position[1]]
        for i in range(ob_rd[1], ob_lu[1]):
            map_i = rd[1] + i - ob_rd[1]
            obs[i][ob_lu[0]:ob_rd[0]] = self.world.DS_state[map_i][lu[0]:rd[0]]
        uav.obs = obs
        return obs
    
    # 获得无人机的奖励
    def get_uav_reward(self):
        count = 0
        mucount = 0
        server_pointList = [
                         (self.servers[0].position[0],self.servers[0].position[1]),
                         (self.servers[1].position[0],self.servers[1].position[1]),
                         (self.servers[2].position[0],self.servers[2].position[1]),
                         (self.servers[3].position[0],self.servers[3].position[1])
                        ]
        uav_pointList = [
                         (self.uavs[0].position[0],self.uavs[0].position[1]),
                         (self.uavs[1].position[0],self.uavs[1].position[1]),
                         (self.uavs[2].position[0],self.uavs[2].position[1]),
                         (self.uavs[3].position[0],self.uavs[3].position[1])
                        ]

        # 遍历每一个传感器
        for sensor in self.sensors:
            # 默认都在于服务器的覆盖范围外
            server_sensor_close = False
            # 默认无人机没有覆盖到传感器
            uav_sensor_close = False
            # 判断是否在服务器的覆盖范围内，会判断四次。这里很巧妙，不可能存在一个传感器存在于多个服务器内
            for i, point in enumerate(server_pointList):
                px, py = point
                server_sensor_distance = np.sqrt((px - sensor.position[0]) ** 2 + (py - sensor.position[1]) ** 2)
                # 四个里面如果有一个存在的话， 则设置为 false
                if server_sensor_distance < self.server_collect_r:
                    server_sensor_close = True
            if(server_sensor_close != True):
                mucount += 1
                for i, point in enumerate(uav_pointList):
                    px, py = point
                    uav_sensor_distance = np.sqrt((px - sensor.position[0]) ** 2 + (py - sensor.position[1]) ** 2)
                    if uav_sensor_distance <= self.uav_collect_r:
                        uav_sensor_close = True
                if(uav_sensor_close):
                    count += 1
        return count/mucount
    
    def get_sensor_obs(self, sensor, if_next_state):
        points = [
            sensor.position,
            self.servers[0].position,
            self.servers[1].position,
            self.servers[2].position,
            self.servers[3].position,
            self.uavs[0].position,
            self.uavs[1].position,
            self.uavs[2].position,
            self.uavs[3].position,
        ]
        distances = []
        # 计算第一个点到其他点之间的距离
        for point in points:
            distance = np.linalg.norm(np.array(points[0]) - np.array(point))
            distances.append(distance)
        # 将结果存储在新的数组中
        self.device_distance = np.array(distances)
    
        if (not if_next_state):
            # 针对没有被任何设备覆盖的传感器，直接跳过
            if (
                self.device_distance[1] > 30 and
                self.device_distance[2] > 30 and
                self.device_distance[3] > 30 and
                self.device_distance[4] > 30 and
                self.device_distance[5] > 40 and
                self.device_distance[6] > 40 and
                self.device_distance[7] > 40 and
                self.device_distance[8] > 40
            ):
                return True, [], [], []
        device_data_amount = [round((sum(sensor.total_data.values())/10000),3), 0, 0, 0, 0, 0, 0, 0, 0]
        device_compute = [0.1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5]
        device_transfer = [0,
                            round(self.world.transmit_rate(self.device_distance[1], sensor),3),
                            round(self.world.transmit_rate(self.device_distance[2], sensor),3),
                            round(self.world.transmit_rate(self.device_distance[3], sensor),3),
                            round(self.world.transmit_rate(self.device_distance[4], sensor),3),
                            round(self.world.transmit_rate(self.device_distance[5], sensor),3),
                            round(self.world.transmit_rate(self.device_distance[6], sensor),3),
                            round(self.world.transmit_rate(self.device_distance[7], sensor),3),
                            round(self.world.transmit_rate(self.device_distance[8], sensor),3)
                            ]
        
        device_transfer = 1 - (rankdata(device_transfer) / len(device_transfer))
        device_transfer[0] = 0

        return False, device_data_amount, device_compute, device_transfer 
    
    #先将 save=False 改为 True，让其先不显示
    def render(self, name=None, epoch=None, save=True):
        plt.figure()
        # 各设备的位置
        for sensor in self.world.sensors:
            sensors = plt.scatter(sensor.position[0], sensor.position[1], c='cornflowerblue', alpha=0.9, label='Sensors', s=sum(sensor.total_data.values())/100)
            plt.annotate(sensor.no + 1, xy=(sensor.position[0], sensor.position[1]), xytext=(sensor.position[0] + 0.1, sensor.position[1] + 0.1))
        for server in self.world.servers:
            servers = plt.scatter(server.position[0], server.position[1], c='orangered', alpha=0.9, label='Servers')
            plt.annotate(server.no + 1, xy=(server.position[0], server.position[1]), xytext=(server.position[0] + 0.1, server.position[1] + 0.1))
            collect_plot = get_circle_plot(server.position, self.server_collect_r)
            plt.fill_between(collect_plot[0], collect_plot[1], collect_plot[2], where=collect_plot[1] > collect_plot[2], color='darkorange', alpha=0.05)
        for uav in self.uavs:
            uavs = plt.scatter(uav.position[0], uav.position[1], c='springgreen', alpha=0.9, label='Uavs')
            plt.annotate(uav.no + 1, xy=(uav.position[0], uav.position[1]), xytext=(uav.position[0] + 0.1, uav.position[1] + 0.1))
            obs_plot = get_circle_plot(uav.position, self.uav_collect_r)
            plt.fill_between(obs_plot[0], obs_plot[1], obs_plot[2], where=obs_plot[1] > obs_plot[2], color='darkgreen', alpha=0.05)
        # 模拟场景车道
        plt.fill_between([0,200], 75, 125, color='navy', alpha=0.2)
        plt.fill_between([75,125], 0, 200, color='navy', alpha=0.2)

        plt.grid()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(handles=[sensors, servers, uavs], labels=['Sensors', 'Servers', 'Uavs'], loc='upper right')
        plt.axis('square')
        plt.xlim([0, self.map_size])
        plt.ylim([0, self.map_size])
        plt.title('all entity position(epoch%s)' % epoch)
        if not save:
            plt.show()
            return
        plt.savefig('%s/%s.png' % (name, epoch))
        plt.close()

    def close(self):
        return None