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
file_path3 = 'logs/go_num_plus/logs_fit_20240109-160306_5.json'
file_path4 = 'logs/go_num_plus/logs_fit_20240107-080237_15.json'
file_path5 = 'logs/go_num_plus/logs_fit_20240108-120619_20.json'
file_path6 = 'logs/go_num_plus/logs_fit_20240111-171902_7.json'
file_path7 = 'logs/go_num_plus/logs_fit_20240113-085213_9.json'

# 加载 JSON 数据
with open(file_path1, 'r') as file:
    data1 = json.load(file)
with open(file_path2, 'r') as file:
    data2 = json.load(file)
with open(file_path3, 'r') as file:
    data3 = json.load(file)
with open(file_path4, 'r') as file:
    data4 = json.load(file)
with open(file_path5, 'r') as file:
    data5 = json.load(file)

# 提取时间戳、步骤数和数值
time1 = [entry[0] for entry in data1]
step_nums1 = [entry[1] for entry in data1]
vals1 = [entry[2] * 1/4 for entry in data1]

time2 = [entry[0] for entry in data2]
step_nums2 = [entry[1] for entry in data2]
# vals2 = [entry[2] for entry in data2]

time3 = [entry[0] for entry in data3]
step_nums3 = [entry[1] for entry in data3]

time4 = [entry[0] for entry in data4]
step_nums4 = [entry[1] for entry in data4]

time5 = [entry[0] for entry in data5]
step_nums5 = [entry[1] for entry in data5]

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

step_nums3 = []
vals3_modified = []
for i, entry in enumerate(data3):
    # if(entry[1] == 2533):
    #     print(i)
    step_nums3.append(entry[1])
    if i >= 177:
        vals3_modified.append(entry[2] * 4/5 * 1/4)
    else:
        vals3_modified.append(entry[2] * 1/4)
    if entry[1] == 13971:
        break

step_nums4 = []
vals4_modified = []
for i, entry in enumerate(data4):
    # if(entry[1] == 5964):
    #     print(i)
    step_nums4.append(entry[1])
    if i >= 324:
        vals4_modified.append(entry[2] * 4/5 * 1/4)
    else:
        vals4_modified.append(entry[2] * 1/4)
    if entry[1] == 13971:
        break

step_nums5 = []
vals5_modified = []
for i, entry in enumerate(data5):
    # if(entry[1] == 5964):
    #     print(i)
    step_nums5.append(entry[1])
    if i >= 383:
        vals5_modified.append(entry[2] * 4/5 * 1/4)
    else:
        vals5_modified.append(entry[2] * 1/4)
    if entry[1] == 13971:
        break

smoothed_vals1 = smooth_data(vals1)
smoothed_vals2 = smooth_data(vals2_modified)
smoothed_vals3 = smooth_data(vals3_modified)
smoothed_vals4 = smooth_data(vals4_modified)
smoothed_vals5 = smooth_data(vals5_modified)

# 每50步进行一次标记
marker_interval = 50

plt.plot(step_nums1[::marker_interval], smoothed_vals1[::marker_interval], linestyle='-', marker='o')
plt.plot(step_nums2[::marker_interval], smoothed_vals2[::marker_interval], linestyle='--', marker='x')
plt.plot(step_nums3[::marker_interval], smoothed_vals3[::marker_interval], linestyle='-.', marker='s')
plt.plot(step_nums4[::marker_interval], smoothed_vals4[::marker_interval], linestyle=':', marker='d')
plt.plot(step_nums5[::marker_interval], smoothed_vals5[::marker_interval], linestyle='-', marker='+')

# plt.xlabel('Step Number')
# plt.ylabel('Reward')
plt.xlabel('执行步数')
plt.ylabel('奖励')
# plt.ylim(1.2, 3.8)
plt.legend(['No-Permanent', 'Permanent-UAV-5', 'Permanent-UAV-10', 'Permanent-UAV-15', 'Permanent-UAV-20'])

# 显示曲线
plt.show()
