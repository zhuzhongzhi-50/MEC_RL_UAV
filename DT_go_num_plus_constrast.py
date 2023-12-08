
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

go_num60_25 = 'logs/go_num_plus/go_num60_25.npy'
go_num80_25 = 'logs/go_num_plus/go_num80_25.npy'

# 提取x和y坐标
go_num60_25 = np.load(go_num60_25)
go_num60_25_x = go_num60_25[:,0]
go_num60_25_y = go_num60_25[:,1]

go_num80_25 = np.load(go_num80_25)
go_num80_25_x = go_num80_25[:,0]
go_num80_25_y = go_num80_25[:,1]

go_num60_25_y_size, go_num60_25_y_counts = np.unique(go_num60_25_y, return_counts=True)
go_num80_25_y_size, go_num80_25_y_counts = np.unique(go_num80_25_y, return_counts=True)

# 对列表进行排序，取前三个最大的数字并计算它们的和
# sorted_go_num20_y_counts = sorted(go_num60_25_y_counts, reverse=True)
# num20_y_top_three = sum(sorted_go_num20_y_counts[:3])

# sorted_go_num25_y_counts = sorted(go_num80_25_y_counts, reverse=True)
# num25_y_top_three = sum(sorted_go_num25_y_counts[:3])


# all_count_x = ['go_num20', 'go_num25', 'go_num30', 'go_num35', 'go_num40', 'go_num45', 'go_num50']
all_count_x = ['60_25', '80_25']

# all_count_y = [sum(go_num20_y_counts), sum(go_num25_y_counts), sum(go_num30_y_counts), sum(go_num35_y_counts), sum(go_num40_y_counts), sum(go_num45_y_counts), sum(go_num50_y_counts)]
all_count_y = [sum(go_num60_25_y_counts), sum(go_num80_25_y_counts)]
max_count_y = [max(go_num60_25_y_counts), max(go_num80_25_y_counts)]
# max_count_y = [num20_y_top_three, num25_y_top_three, num30_y_top_three, num35_y_top_three, num40_y_top_three, num45_y_top_three, num50_y_top_three]


width = 0.4
color = 'white'
hatch_pattern = '//'
edgecolor = 'black'
linewidth = 1.5

x = np.arange(len(all_count_x))  # 生成每个柱的中心坐标

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, all_count_y, width, color=color, edgecolor=edgecolor, linewidth=linewidth)
bars2 = ax.bar(x + width/2, max_count_y, width, color=color, edgecolor=edgecolor, linewidth=linewidth)

for bar in bars1:
    bar.set_hatch(hatch_pattern)

for bar in bars2:
    bar.set_hatch(hatch_pattern)

ax.set_xticks(x)
ax.set_xticklabels(all_count_x)
ax.set_xlabel('Dispatch Threshold')
ax.set_ylabel('The number of UAV dispatch times')
ax.legend(['All Count', 'Max Three Count'])
plt.show()

