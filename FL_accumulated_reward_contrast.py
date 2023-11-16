import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def smooth_data(data, weight=0.9):
    smoothed_data = []
    last = data[0]
    for point in data:
        smoothed_point = last * weight + (1 - weight) * point
        smoothed_data.append(smoothed_point)
        last = smoothed_point
    return smoothed_data

# 导入从 TensorBoard 导出的数据文件
file_path1 = 'logs/tensorboard_result/reward_0.json'
file_path2 = 'logs/tensorboard_result/reward_0.25.json'
file_path3 = 'logs/tensorboard_result/reward_0.5.json'
file_path4 = 'logs/tensorboard_result/reward_0.75.json'
file_path5 = 'logs/tensorboard_result/reward_1.json'

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
vals1 = [entry[2] for entry in data1]
vals1 = np.array(vals1) / 2

time2 = [entry[0] for entry in data2]
step_nums2 = [entry[1] for entry in data2]
vals2 = [entry[2] for entry in data2]
vals2 = np.array(vals2) / 2

time3 = [entry[0] for entry in data3]
step_nums3 = [entry[1] for entry in data3]
vals3 = [entry[2] for entry in data3]
vals3 = np.array(vals3) / 2

time4 = [entry[0] for entry in data4]
step_nums4 = [entry[1] for entry in data4]
vals4 = [entry[2] for entry in data4]
vals4 = np.array(vals4) / 2

time5 = [entry[0] for entry in data5]
step_nums5 = [entry[1] for entry in data5]
vals5 = [entry[2] for entry in data5]
vals5 = np.array(vals5) / 2

smoothed_vals1 = smooth_data(vals1)
smoothed_vals2 = smooth_data(vals2)
smoothed_vals3 = smooth_data(vals3)
smoothed_vals4 = smooth_data(vals4)
smoothed_vals5 = smooth_data(vals5)

plt.plot(step_nums1, smoothed_vals1)
plt.plot(step_nums2, smoothed_vals2)
plt.plot(step_nums3, smoothed_vals3)
plt.plot(step_nums4, smoothed_vals4)
plt.plot(step_nums5, smoothed_vals5)

plt.xlabel('Step Number')
plt.ylabel('Accumulated Reward')
plt.legend(['FL 0', 'FL 0.25', 'FL 0.5', 'FL 0.75', 'FL 1'])

# 显示曲线
plt.show()


