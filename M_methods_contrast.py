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

# 导入从 TensorBoard 导出的数据文件
# file_path1 = 'logs/methods/1.json'
# file_path2 = 'logs/methods/2.json'
# file_path3 = 'logs/methods/3.json'
# file_path4 = 'logs/methods/4.json'
# file_path5 = 'logs/methods/5.json'
file_path1 = 'logs/methods/methods_Fixed_ISODATA.json'
file_path2 = 'logs/methods/methods_AC.json'
file_path3 = 'logs/methods/methods_ACPC.json'
file_path4 = 'logs/methods/methods_ACPCRD.json'

# 加载 JSON 数据
with open(file_path1, 'r') as file:
    data1 = json.load(file)
with open(file_path2, 'r') as file:
    data2 = json.load(file)
with open(file_path3, 'r') as file:
    data3 = json.load(file)
with open(file_path4, 'r') as file:
    data4 = json.load(file)
# with open(file_path5, 'r') as file:
#     data5 = json.load(file)

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

# time5 = [entry[0] for entry in data5]
# step_nums5 = [entry[1] for entry in data5]
# vals5 = [entry[2] for entry in data5]

smoothed_vals1 = smooth_data(vals1)
smoothed_vals2 = smooth_data(vals2)
smoothed_vals3 = smooth_data(vals3)
smoothed_vals4 = smooth_data(vals4)
# smoothed_vals5 = smooth_data(vals5)

# 每100步进行一次标记
marker_interval = 50

plt.plot(step_nums1[::marker_interval], smoothed_vals1[::marker_interval], linestyle='-', marker='o')
plt.plot(step_nums2[::marker_interval], smoothed_vals2[::marker_interval], linestyle='--', marker='x')
plt.plot(step_nums3[::marker_interval], smoothed_vals3[::marker_interval], linestyle='-.', marker='s')
plt.plot(step_nums4[::marker_interval], smoothed_vals4[::marker_interval], linestyle=':', marker='d')
# plt.plot(step_nums5[::marker_interval], smoothed_vals5[::marker_interval], linestyle='-', marker='+')


# plt.xlabel('Step Number')
# plt.ylabel('Reward')
plt.xlabel('执行步数')
plt.ylabel('奖励')
# plt.ylim(1.2, 3.8)
plt.legend(['Fixed ISODATA', 'AC', 'PCAC', 'PCRDAC'])

# 显示曲线
plt.show()
