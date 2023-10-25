import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# plt.figure(figsize=(10, 6))  # 增大图形的宽度

# 导入从 TensorBoard 导出的数据文件
file_path1 = 'logs/tensorboard_result/FL_1.json'
file_path2 = 'logs/tensorboard_result/FL_0.75.json'
file_path3 = 'logs/tensorboard_result/FL_0.5.json'
file_path4 = 'logs/tensorboard_result/FL_0.25.json'
file_path5 = 'logs/tensorboard_result/FL_0.json'

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

# 将所有数据放入一个列表中
all_data = [vals1, vals2, vals3, vals4, vals5]

# 设置每个箱线图的位置
positions = [1, 1.5, 2, 2.5, 3]

# 设置每个箱线图的颜色
colors = ['#2f77af', '#f9722b', '#3b9d37', '#ca0b24', '#8457b1']

# 设置每个箱线图的填充颜色
facecolors = ['#bbd7ed', '#fdd5bf', '#bee8bd', '#faaab5', '#dacde8']

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
plt.xticks(positions, ['FL 0', 'FL 0.25', 'FL 0.5', 'FL 0.75', 'FL 1'])
# 添加标题和标签
# plt.title('Multiple Boxplots')
plt.xlabel('Edge Synergy Joint Factor')
plt.ylabel('Coverage Rate')

# 显示图形
plt.show()
