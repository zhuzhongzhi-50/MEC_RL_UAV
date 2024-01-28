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

# 设置使用LaTeX渲染文本
# plt.rc('text', usetex=True)


def smooth_data(data, weight=0.99):
    smoothed_data = []
    last = data[0]
    for point in data:
        smoothed_point = last * weight + (1 - weight) * point
        smoothed_data.append(smoothed_point)
        last = smoothed_point
    return smoothed_data

# 导入从 TensorBoard 导出的数据文件

file_path1 = 'logs/peak_age/peak_age_0.5.json'
file_path2 = 'logs/peak_age/peak_age_1.json'
file_path3 = 'logs/peak_age/peak_age_DT_30.json'
file_path4 = 'logs/peak_age/peak_age_DT_25.json'

# 加载 JSON 数据
with open(file_path1, 'r') as file:
    data1 = json.load(file)
with open(file_path2, 'r') as file:
    data2 = json.load(file)
with open(file_path3, 'r') as file:
    data3 = json.load(file)
with open(file_path4, 'r') as file:
    data4 = json.load(file)

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

smoothed_vals1 = smooth_data(vals1)
smoothed_vals2 = smooth_data(vals2)
smoothed_vals3 = smooth_data(vals3)
smoothed_vals4 = smooth_data(vals4)

# 每50步进行一次标记
marker_interval = 50

plt.plot(step_nums1[::marker_interval], smoothed_vals1[::marker_interval], linestyle='-', marker='o')
plt.plot(step_nums2[::marker_interval], smoothed_vals2[::marker_interval], linestyle='--', marker='x')
plt.plot(step_nums3[::marker_interval], smoothed_vals3[::marker_interval], linestyle='-.', marker='s')

# plt.xlabel('Step Number')
# plt.ylabel('Peak Age')
plt.xlabel('执行步数')
plt.ylabel('峰值年龄')
# plt.ylim(1.2, 3.8)
# plt.legend(['\u03BC = 0', '\u03BC = 1', '\u03BC = 0.5 \u03C8 = 30', '\u03BC = 0.5 \u03C8 = 25'])
plt.legend(['\u03BC = 0.5', '\u03BC = 0.5 \u03C8 = 30', '\u03BC = 0.5 \u03C8 = 25'])

# 显示曲线
plt.show()
