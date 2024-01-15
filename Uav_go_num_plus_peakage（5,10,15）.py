import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from matplotlib import rcParams

config = {
    "font.family":'serif',
    "mathtext.fontset":'stix',
    "font.serif":['SimSun']
}
rcParams.update(config)

def smooth_data(data, weight=0.95):
    smoothed_data = []
    last = data[0]
    for point in data:
        smoothed_point = last * weight + (1 - weight) * point
        smoothed_data.append(smoothed_point)
        last = smoothed_point
    return smoothed_data

# 导入从 TensorBoard 导出的数据文件
# file_path1 = 'logs/go_num_plus/reward_60_peakage_no.json'
file_path1 = 'logs/go_num_plus/reward_60_peakage_no_15000.json'
file_path2 = 'logs/go_num_plus/logs_fit_20240109-160306（peakage_5）.json'
file_path3 = 'logs/go_num_plus/reward_60_peakage_uav1_15000.json'
file_path4 = 'logs/go_num_plus/logs_fit_20240107-080237（peakage_15）.json'

# 加载 JSON 数据
with open(file_path1, 'r') as file:
    data1 = json.load(file)
with open(file_path2, 'r') as file:
    data2 = json.load(file)
with open(file_path3, 'r') as file:
    data3 = json.load(file)
with open(file_path4, 'r') as file:
    data4 = json.load(file)

print(data1)

# 11009

# 提取时间戳、步骤数和数值
time1 = [entry[0] for entry in data1]
step_nums1 = [entry[1] for entry in data1]
vals1 = [entry[2] for entry in data1]

time2 = [entry[0] for entry in data2]
step_nums2 = [entry[1] for entry in data2]
vals2 = [entry[2] for entry in data2]

time3 = [entry[0] for entry in data3]
step_nums3 = [entry[1] for entry in data3]
vals3 = [entry[2] for entry in data3]

time4 = [entry[0] for entry in data4]
step_nums4 = [entry[1] for entry in data4]
vals4 = [entry[2] for entry in data4]

# 13999
vals1 = []
step_nums1 = []
for i, entry in enumerate(data1):
    step_nums1.append(entry[1])
    vals1.append(entry[2])
    if entry[1] == 13095:
        break

# 11009

vals2 = []
step_nums2 = []
for i, entry in enumerate(data2):
    step_nums2.append(entry[1])
    vals2.append(entry[2])
    if entry[1] == 13095:
        break


vals3 = []
step_nums3 = []
for i, entry in enumerate(data3):
    step_nums3.append(entry[1])
    vals3.append(entry[2])
    if entry[1] == 13095:
        break


vals4 = []
step_nums4 = []
for i, entry in enumerate(data4):
    step_nums4.append(entry[1])
    vals4.append(entry[2])
    if entry[1] == 13095:
        break

smoothed_vals1 = smooth_data(vals1)
smoothed_vals2 = smooth_data(vals2)
smoothed_vals3 = smooth_data(vals3)
smoothed_vals4 = smooth_data(vals4)

# 每50步进行一次标记
marker_interval = 50

plt.plot(step_nums1[::marker_interval], smoothed_vals1[::marker_interval], linestyle='-', marker='o')
plt.plot(step_nums2[::marker_interval], smoothed_vals2[::marker_interval], linestyle='--', marker='x')
plt.plot(step_nums3[::marker_interval], smoothed_vals3[::marker_interval], linestyle='-.', marker='s')
plt.plot(step_nums4[::marker_interval], smoothed_vals4[::marker_interval], linestyle=':', marker='d')

plt.xlabel('执行步数')
plt.ylabel('峰值年龄')
# plt.ylim(10, 25)
# plt.legend(['No-Permanent', 'Permanent-UAV-5', 'Permanent-UAV-10', 'Permanent-UAV-15'])
plt.legend(['无常驻', '常驻-5', '常驻-10', '常驻-15'])

# 显示曲线
plt.show()
