import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def smooth_data(data, weight=0.95):
    smoothed_data = []
    last = data[0]
    for point in data:
        smoothed_point = last * weight + (1 - weight) * point
        smoothed_data.append(smoothed_point)
        last = smoothed_point
    return smoothed_data

# 导入从 TensorBoard 导出的数据文件

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

# 将所有数据放入一个列表中
all_data = [vals1, vals2, vals3, vals4, vals5, vals6, vals7]

# 设置每个箱线图的位置
positions = [1, 1.5, 2, 2.5, 3, 3.5, 4]

# 设置每个箱线图的颜色
colors = ['#2f77af', '#f9722b', '#3b9d37', '#ca0b24', '#8457b1', '#8c564b', '#e377c2']

# 设置每个箱线图的填充颜色
facecolors = ['#bbd7ed', '#fdd5bf', '#bee8bd', '#faaab5', '#dacde8', '#d6b8b2', '#f4c9e7']

for pos, data, color, facecolor in zip(positions, all_data, colors, facecolors):
    # 设置箱体、须等属性
    boxprops = {'color': color, 'linewidth': 2}  # 设置箱子的颜色
    whiskerprops = {'color': 'black', 'linestyle': '--'}
    medianprops = {'color': color, 'linewidth': 2}
    capprops = {'color': 'black'}

    # 绘制箱线图
    box_plot = plt.boxplot(
        data,
        positions=[pos],
        showfliers=False,  # 不显示异常值
        patch_artist=True,  # 允许填充
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        medianprops=medianprops,
        capprops=capprops
    )

    # 设置箱体的填充颜色
    for patch in box_plot['boxes']:
        patch.set_facecolor(facecolor)
        
# 设置 x 轴刻度标签
plt.xticks(positions, ['0', '2', '4', '6', '8', '10', '12'])
# 添加标题和标签
# plt.title('Multiple Boxplots')
plt.xlabel('Update Frequency')
plt.ylabel('Reward Statistic')

# 显示图形
plt.show()
