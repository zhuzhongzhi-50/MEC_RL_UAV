import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def smooth_data(data, weight=0.999):
    smoothed_data = []
    last = data[0]
    for point in data:
        smoothed_point = last * weight + (1 - weight) * point
        smoothed_data.append(smoothed_point)
        last = smoothed_point
    return smoothed_data

# 导入从 TensorBoard 导出的数据文件
# file_path1 = 'logs/update_frequency/frequency_0_1.json'
# file_path2 = 'logs/update_frequency/frequency_0_2.json'
# file_path3 = 'logs/update_frequency/frequency_0_4.json'
# file_path4 = 'logs/update_frequency/frequency_0_6.json'
# file_path5 = 'logs/update_frequency/frequency_0_8.json'
# file_path6 = 'logs/update_frequency/frequency_0_10.json'
# file_path7 = 'logs/update_frequency/frequency_0_12.json'

file_path1 = 'logs/update_frequency/frequency_1_0.json'
file_path2 = 'logs/update_frequency/frequency_0_2.json'
file_path3 = 'logs/update_frequency/frequency_0_4.json'
file_path4 = 'logs/update_frequency/frequency_0_6.json'
file_path5 = 'logs/update_frequency/frequency_0_8.json'
file_path6 = 'logs/update_frequency/frequency_0_10.json'
file_path7 = 'logs/update_frequency/frequency_0_12.json'

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
with open(file_path6, 'r') as file:
    data6 = json.load(file)
with open(file_path7, 'r') as file:
    data7 = json.load(file)

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

time5 = [entry[0] for entry in data5]
step_nums5 = [entry[1] for entry in data5]
vals5 = [entry[2] for entry in data5]

time6 = [entry[0] for entry in data6]
step_nums6 = [entry[1] for entry in data6]
vals6 = [entry[2] for entry in data6]

time7 = [entry[0] for entry in data7]
step_nums7 = [entry[1] for entry in data7]
vals7 = [entry[2] for entry in data7]

smoothed_vals1 = smooth_data(vals1)
smoothed_vals2 = smooth_data(vals2)
smoothed_vals3 = smooth_data(vals3)
smoothed_vals4 = smooth_data(vals4)
smoothed_vals5 = smooth_data(vals5)
smoothed_vals6 = smooth_data(vals6)
smoothed_vals7 = smooth_data(vals7)

plt.plot(step_nums1, smoothed_vals1)
plt.plot(step_nums2, smoothed_vals2)
plt.plot(step_nums3, smoothed_vals3)
plt.plot(step_nums4, smoothed_vals4)
plt.plot(step_nums5, smoothed_vals5)
plt.plot(step_nums6, smoothed_vals6)
plt.plot(step_nums7, smoothed_vals7)

plt.xlabel('Step Number')
plt.ylabel('Reward')
plt.ylim(1.5, 3.8)
plt.legend(['0', '2', '4', '6', '8', '10', '12'])
# plt.legend(['0', '4', '8', '12'])


# 显示曲线
plt.show()
