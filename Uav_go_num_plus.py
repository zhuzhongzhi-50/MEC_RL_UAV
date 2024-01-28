import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from matplotlib import rcParams

config = {
    "font.family":'serif',
    # "font.size": 20,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)

def smooth_data(data, weight=0.99):
    smoothed_data = []
    last = data[0]
    for point in data:
        smoothed_point = last * weight + (1 - weight) * point
        smoothed_data.append(smoothed_point)
        last = smoothed_point
    return smoothed_data

# file_path1 = 'logs/go_num_plus/reward_60_no.json'
# file_path2 = 'logs/go_num_plus/reward_60_uav1.json'

file_path1 = 'logs/go_num_plus/reward_60_no_15000.json'
file_path2 = 'logs/go_num_plus/reward_60_uav1_15000.json'

# 加载 JSON 数据
with open(file_path1, 'r') as file:
    data1 = json.load(file)
with open(file_path2, 'r') as file:
    data2 = json.load(file)

print(data2)

# 提取时间戳、步骤数和数值
time1 = [entry[0] for entry in data1]
step_nums1 = [entry[1] for entry in data1]
vals1 = [entry[2] * 1/4 for entry in data1]

time2 = [entry[0] for entry in data2]
step_nums2 = [entry[1] for entry in data2]
# vals2 = [entry[2] for entry in data2]

# 4183  
vals1 = []
step_nums1 = []
for i, entry in enumerate(data1):
    step_nums1.append(entry[1])
    vals1.append(entry[2] * 1/4)
    if entry[1] == 13999:
        break

step_nums2 = []
vals2_modified = []
for i, entry in enumerate(data2):
    # if(entry[1] == 4183):
    #     print(i)
    step_nums2.append(entry[1])
    if i >= 266:
        vals2_modified.append(entry[2] * 4/5 * 1/4)
    else:
        vals2_modified.append(entry[2] * 1/4)
    if entry[1] == 13971:
        break

smoothed_vals1 = smooth_data(vals1)
smoothed_vals2 = smooth_data(vals2_modified)

# 每50步进行一次标记
marker_interval = 50

plt.plot(step_nums1[::marker_interval], smoothed_vals1[::marker_interval], linestyle='-', marker='o')
plt.plot(step_nums2[::marker_interval], smoothed_vals2[::marker_interval], linestyle='--', marker='x')

# plt.xlabel('Step Number')
# plt.ylabel('Reward')
plt.xlabel('执行步数')
plt.ylabel('奖励')
# plt.ylim(1.2, 3.8)
plt.legend(['No-Permanent', 'Permanent-UAV'])

# 显示曲线
plt.show()
