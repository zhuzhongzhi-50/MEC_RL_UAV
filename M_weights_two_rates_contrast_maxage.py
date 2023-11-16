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
file_path8 = 'logs/weights_two_rates_maxage/weights_two_rates_maxage_1.0_0.0.json'
file_path5 = 'logs/weights_two_rates_maxage/weights_two_rates_maxage_0.5_0.5.json'
file_path1 = 'logs/weights_two_rates_maxage/weights_two_rates_maxage_0.0_1.0.json'
file_path6 = 'logs/weights_two_rates_maxage/weights_two_rates_maxage_0.7_0.3.json'
file_path4 = 'logs/weights_two_rates_maxage/weights_two_rates_maxage_0.3_0.7.json'
file_path2 = 'logs/weights_two_rates_maxage/weights_two_rates_maxage_0.1_0.9.json'
file_path7 = 'logs/weights_two_rates_maxage/weights_two_rates_maxage_0.9_0.1.json'
file_path3 = 'logs/weights_two_rates_maxage/weights_two_rates_maxage_0.2_0.8.json'

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
with open(file_path8, 'r') as file:
    data8 = json.load(file)

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

time8 = [entry[0] for entry in data8]
step_nums8 = [entry[1] for entry in data8]
vals8 = [entry[2] for entry in data8]

smoothed_vals1 = smooth_data(vals1)
smoothed_vals2 = smooth_data(vals2)
smoothed_vals3 = smooth_data(vals3)
smoothed_vals4 = smooth_data(vals4)
smoothed_vals5 = smooth_data(vals5)
smoothed_vals6 = smooth_data(vals6)
smoothed_vals7 = smooth_data(vals7)
smoothed_vals8 = smooth_data(vals8)

plt.plot(step_nums1, smoothed_vals1)
plt.plot(step_nums2, smoothed_vals2)
plt.plot(step_nums3, smoothed_vals3)
plt.plot(step_nums4, smoothed_vals4)
plt.plot(step_nums5, smoothed_vals5)
plt.plot(step_nums6, smoothed_vals6)
plt.plot(step_nums7, smoothed_vals7)
plt.plot(step_nums8, smoothed_vals8)

plt.xlabel('Step Number')
plt.ylabel('MaxAge')
# plt.ylim(1.2, 3.8)
plt.legend(['0.0', '0.1', '0.2', '0.3', '0.5', '0.7', '0.9', '1.0'])

# 显示曲线
plt.show()
