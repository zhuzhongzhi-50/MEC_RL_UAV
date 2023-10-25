import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def smooth_data(data, weight=0.99):
    smoothed_data = []
    last = data[0]
    for point in data:
        smoothed_point = last * weight + (1 - weight) * point
        smoothed_data.append(smoothed_point)
        last = smoothed_point
    return smoothed_data

# 导入从 TensorBoard 导出的数据文件
file_path1 = 'logs/age/worst_age_0_8.json'

# 加载 JSON 数据
with open(file_path1, 'r') as file:
    data1 = json.load(file)

# 提取时间戳、步骤数和数值
time1 = [entry[0] for entry in data1]
step_nums1 = [entry[1] for entry in data1]
vals1 = [entry[2] for entry in data1]

smoothed_vals1 = smooth_data(vals1)

plt.plot(step_nums1, smoothed_vals1)

# 移动平均算法并不好用
# window_size = 100
# smoothed_vals1 = np.convolve(vals1, np.ones(window_size)/window_size, mode='valid')

# plt.plot(step_nums1[window_size-1:], smoothed_vals1)

plt.xlabel('Step')
plt.ylabel('Worst Sensor Age')
# plt.ylim(1.2, 3.8)
# plt.legend(['1', '4', '6', '8', '10'])

# 显示曲线
plt.show()
