import matplotlib.pyplot as plt
import numpy as np

def smooth_data(data, weight=0.98):
    smoothed_data = []
    last = data[0]
    for point in data:
        smoothed_point = last * weight + (1 - weight) * point
        smoothed_data.append(smoothed_point)
        last = smoothed_point
    return smoothed_data

# 指定要加载的 .npy 文件路径
uav_positoin_x0 = 'logs/position/uav_x0.npy'
uav_positoin_y0 = 'logs/position/uav_y0.npy'
uav_positoin_x1 = 'logs/position/uav_x1.npy'
uav_positoin_y1 = 'logs/position/uav_y1.npy'
uav_positoin_x2 = 'logs/position/uav_x2.npy'
uav_positoin_y2 = 'logs/position/uav_y2.npy'
uav_positoin_x3 = 'logs/position/uav_x3.npy'
uav_positoin_y3 = 'logs/position/uav_y3.npy'

# 加载 .npy 文件
x0 = np.load(uav_positoin_x0)
y0 = np.load(uav_positoin_y0)
x1 = np.load(uav_positoin_x1)
y1 = np.load(uav_positoin_y1)
x2 = np.load(uav_positoin_x2)
y2 = np.load(uav_positoin_y2)
x3 = np.load(uav_positoin_x3)
y3 = np.load(uav_positoin_y3)

# 计算两两无人机之间的距离和
dist_sums = []
for i in range(1000):
    dist_01 = np.sqrt((x0[i] - x1[i])**2 + (y0[i] - y1[i])**2)
    dist_02 = np.sqrt((x0[i] - x2[i])**2 + (y0[i] - y2[i])**2)
    dist_03 = np.sqrt((x0[i] - x3[i])**2 + (y0[i] - y3[i])**2)
    dist_12 = np.sqrt((x1[i] - x2[i])**2 + (y1[i] - y2[i])**2)
    dist_13 = np.sqrt((x1[i] - x3[i])**2 + (y1[i] - y3[i])**2)
    dist_23 = np.sqrt((x2[i] - x3[i])**2 + (y2[i] - y3[i])**2)
    # dist_sum = np.mean([dist_01, dist_02, dist_03, dist_12, dist_13, dist_23], axis=0)
    # dist_sums.append(dist_sum)

    distances = [dist_01, dist_02, dist_03, dist_12, dist_13, dist_23]

    # max1 = max(distances)
    # distances.remove(max1)

    # max2 = max(distances)
    # distances.remove(max2)

    mean = np.mean(distances)

    dist_sums.append(mean)

mean_dist_sums = np.mean(dist_sums)

smoothed_vals1 = smooth_data(dist_sums)

# 指定要加载的 .npy 文件路径
uav_positoin_x0_last = 'logs/position/uav_x0_last.npy'
uav_positoin_y0_last = 'logs/position/uav_y0_last.npy'
uav_positoin_x1_last = 'logs/position/uav_x1_last.npy'
uav_positoin_y1_last = 'logs/position/uav_y1_last.npy'
uav_positoin_x2_last = 'logs/position/uav_x2_last.npy'
uav_positoin_y2_last = 'logs/position/uav_y2_last.npy'
uav_positoin_x3_last = 'logs/position/uav_x3_last.npy'
uav_positoin_y3_last = 'logs/position/uav_y3_last.npy'

# 加载 .npy 文件
x0_last = np.load(uav_positoin_x0_last)
y0_last = np.load(uav_positoin_y0_last)
x1_last = np.load(uav_positoin_x1_last)
y1_last = np.load(uav_positoin_y1_last)
x2_last = np.load(uav_positoin_x2_last)
y2_last = np.load(uav_positoin_y2_last)
x3_last = np.load(uav_positoin_x3_last)
y3_last = np.load(uav_positoin_y3_last)

# 计算两两无人机之间的距离和
dist_sums_last = []
for i in range(1000):
    dist_01_last = np.sqrt((x0_last[1000+i] - x1_last[1000+i])**2 + (y0_last[1000+i] - y1_last[1000+i])**2)
    dist_02_last = np.sqrt((x0_last[1000+i] - x2_last[1000+i])**2 + (y0_last[1000+i] - y2_last[1000+i])**2)
    dist_03_last = np.sqrt((x0_last[1000+i] - x3_last[1000+i])**2 + (y0_last[1000+i] - y3_last[1000+i])**2)
    dist_12_last = np.sqrt((x1_last[1000+i] - x2_last[1000+i])**2 + (y1_last[1000+i] - y2_last[1000+i])**2)
    dist_13_last = np.sqrt((x1_last[1000+i] - x3_last[1000+i])**2 + (y1_last[1000+i] - y3_last[1000+i])**2)
    dist_23_last = np.sqrt((x2_last[1000+i] - x3_last[1000+i])**2 + (y2_last[1000+i] - y3_last[1000+i])**2)
    # dist_sum_last = dist_01_last + dist_02_last + dist_03_last + dist_12_last + dist_13_last + dist_23_last
    # dist_sum_last = np.mean([dist_01_last, dist_02_last, dist_03_last, dist_12_last, dist_13_last, dist_23_last], axis=0)
    distances = [dist_01_last, dist_02_last, dist_03_last, dist_12_last, dist_13_last, dist_23_last]

    # max1 = max(distances)
    # distances.remove(max1)

    # max2 = max(distances)
    # distances.remove(max2)

    mean = np.mean(distances)

    dist_sums_last.append(mean)

mean_dist_sums_last = np.mean(dist_sums_last)

smoothed_vals2 = smooth_data(dist_sums_last)

# # 创建图形对象和子图
# fig, ax = plt.subplots()

# # 绘制距离和随时间变化的图表
# ax.plot(range(1, len(smoothed_vals1)+1), smoothed_vals1, label='First 1000 Step')
# # 绘制平均线
# # ax.axhline(mean_dist_sums, color='b', linestyle='--', alpha=0.3, label='First 1000 Mean Distance')
# # ax.axhline(mean_dist_sums, color='b', linestyle='--', alpha=0.3)

# ax.plot(range(1, len(smoothed_vals2)+1), smoothed_vals2, label='Last 1000 Step')
# # ax.axhline(mean_dist_sums_last, color='r', linestyle='--', alpha=0.3, label='Last 1000 Mean Distance')
# ax.axhline(mean_dist_sums_last, color='r', linestyle='--', alpha=0.3)

# plt.xlabel('Step')
# plt.ylabel('Uavs Distance')
# plt.ylim(60, 140)
# # plt.ylim(65, 105)
# # plt.title('Distance Sum between Drones')
# plt.legend()
# plt.show()


fig, ax = plt.subplots()

# 对第一条线进行设置
ax.plot(range(1, len(smoothed_vals1)+1), smoothed_vals1, label='First 1000 Step', linestyle='-', marker='o', markevery=50)

# 对第二条线进行设置
ax.plot(range(1, len(smoothed_vals2)+1), smoothed_vals2, label='Last 1000 Step', linestyle='--', marker='x', markevery=50)

ax.axhline(mean_dist_sums_last, color='r', linestyle='--', alpha=0.3)

plt.xlabel('Step')
plt.ylabel('Uavs Distance')
plt.ylim(60, 140)
plt.legend()
plt.show()