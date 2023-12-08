
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

go_num20 = 'logs/go_num/go_num20.npy'
go_num25 = 'logs/go_num/go_num25.npy'
go_num30 = 'logs/go_num/go_num30.npy'
go_num35 = 'logs/go_num/go_num34.npy'
go_num40 = 'logs/go_num/go_num40.npy'
go_num45 = 'logs/go_num/go_num45.npy'
go_num50 = 'logs/go_num/go_num50.npy'

# go_num25_UAV_no_move = 'logs/go_num/go_num25_UAV_no_move.npy'
# go_num25_UAV_no_move = np.load(go_num25_UAV_no_move)
# go_num25_UAV_no_move_x = go_num25_UAV_no_move[:,0]
# go_num25_UAV_no_move_y = go_num25_UAV_no_move[:,1]
# go_num25_UAV_no_move_y_size, go_num25_UAV_no_move_y_counts = np.unique(go_num25_UAV_no_move_y, return_counts=True)
# sorted_go_num25_UAV_no_move_y_counts = sorted(go_num25_UAV_no_move_y_counts, reverse=True)
# num25_UAV_no_move_y_top_three = sum(sorted_go_num25_UAV_no_move_y_counts[:3])

# 提取x和y坐标
go_num20 = np.load(go_num20)
go_num20_x = go_num20[:,0]
go_num20_y = go_num20[:,1]
print(len(go_num20_y))

go_num25 = np.load(go_num25)
go_num25_x = go_num25[:,0]
go_num25_y = go_num25[:,1]

go_num30 = np.load(go_num30)
go_num30_x = go_num30[:,0]
go_num30_y = go_num30[:,1]

go_num35 = np.load(go_num35)
go_num35_x = go_num35[:,0]
go_num35_y = go_num35[:,1]
print(len(go_num35_y))

go_num40 = np.load(go_num40)
go_num40_x = go_num40[:,0]
go_num40_y = go_num40[:,1]

go_num45 = np.load(go_num45)
go_num45_x = go_num45[:,0]
go_num45_y = go_num45[:,1]

go_num50 = np.load(go_num50)
go_num50_x = go_num50[:,0]
go_num50_y = go_num50[:,1]
print(len(go_num50_y))

# 统计y坐标的出现次数
go_num20_y_size, go_num20_y_counts = np.unique(go_num20_y, return_counts=True)
go_num25_y_size, go_num25_y_counts = np.unique(go_num25_y, return_counts=True)
go_num30_y_size, go_num30_y_counts = np.unique(go_num30_y, return_counts=True)
go_num35_y_size, go_num35_y_counts = np.unique(go_num35_y, return_counts=True)
go_num40_y_size, go_num40_y_counts = np.unique(go_num40_y, return_counts=True)
go_num45_y_size, go_num45_y_counts = np.unique(go_num45_y, return_counts=True)
go_num50_y_size, go_num50_y_counts = np.unique(go_num50_y, return_counts=True)

# 对列表进行排序，取前三个最大的数字并计算它们的和
sorted_go_num20_y_counts = sorted(go_num20_y_counts, reverse=True)
num20_y_top_three = sum(sorted_go_num20_y_counts[:3])

sorted_go_num25_y_counts = sorted(go_num25_y_counts, reverse=True)
num25_y_top_three = sum(sorted_go_num25_y_counts[:3])

sorted_go_num30_y_counts = sorted(go_num30_y_counts, reverse=True)
num30_y_top_three = sum(sorted_go_num30_y_counts[:3])

sorted_go_num35_y_counts = sorted(go_num35_y_counts, reverse=True)
num35_y_top_three = sum(sorted_go_num35_y_counts[:3])

sorted_go_num40_y_counts = sorted(go_num40_y_counts, reverse=True)
num40_y_top_three = sum(sorted_go_num40_y_counts[:3])

sorted_go_num45_y_counts = sorted(go_num45_y_counts, reverse=True)
num45_y_top_three = sum(sorted_go_num45_y_counts[:3])

sorted_go_num50_y_counts = sorted(go_num50_y_counts, reverse=True)
num50_y_top_three = sum(sorted_go_num50_y_counts[:3])


# 柱状图的 x 坐标位置
# all_count_x = ['go_num20', 'go_num25', 'go_num30', 'go_num35', 'go_num40', 'go_num45', 'go_num50']
all_count_x = ['20', '25', '30', '35', '40', '45', '50']

# all_count_y = [sum(go_num20_y_counts), sum(go_num25_y_counts), sum(go_num30_y_counts), sum(go_num35_y_counts), sum(go_num40_y_counts), sum(go_num45_y_counts), sum(go_num50_y_counts)]
all_count_y = [sum(go_num20_y_counts), sum(go_num25_y_counts), sum(go_num30_y_counts), sum(go_num35_y_counts), sum(go_num40_y_counts), sum(go_num45_y_counts), sum(go_num50_y_counts)]
# max_count_y = [max(go_num20_y_counts), max(go_num25_y_counts), max(go_num30_y_counts), max(go_num35_y_counts), max(go_num40_y_counts), max(go_num45_y_counts), max(go_num50_y_counts)]
max_count_y = [num20_y_top_three, num25_y_top_three, num30_y_top_three, num35_y_top_three, num40_y_top_three, num45_y_top_three, num50_y_top_three]


width = 0.4
color = 'white'
hatch_pattern = '//'
edgecolor = 'black'
linewidth = 1.5

x = np.arange(len(all_count_x))  # 生成每个柱的中心坐标

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, all_count_y, width, color=color, edgecolor=edgecolor, linewidth=linewidth)
# bars2 = ax.bar(x + width/2, max_count_y, width, color=color, edgecolor=edgecolor, linewidth=linewidth)

for bar in bars1:
    bar.set_hatch(hatch_pattern)

# for bar in bars2:
#     bar.set_hatch(hatch_pattern)

ax.set_xticks(x)
ax.set_xticklabels(all_count_x)
ax.set_xlabel('Dispatch Threshold')
ax.set_ylabel('The number of UAV dispatch times')
# ax.legend(['All Count', 'Max Three Count'])
plt.show()

